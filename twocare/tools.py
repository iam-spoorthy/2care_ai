"""Tools exposed to the agent, and the guards that hold no matter what the model says.

Hard guarantees live here, not in the prompt: a chart is opened only with phone
plus date of birth, verification locks after repeated misses, a patient can only
see and change their own appointments, and only real open slots can be booked.
Behaviour that needs judgement (confirming before a change, respecting "not
yet", refusing medical advice) lives in the prompt and is what the eval scores.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field

from .clinic import WEEKDAYS, Clinic, Ticket, parse_slot_id, slot_id

MAX_VERIFY_ATTEMPTS = 3
MAX_SLOTS_RETURNED = 8

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "verify_patient",
            "description": "Open the caller's chart. Requires both the phone number on file and the date of birth. Must succeed before any appointment can be viewed or changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string", "description": "Phone number as the caller said it."},
                    "date_of_birth": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["phone", "date_of_birth"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_providers",
            "description": "List the clinic's doctors, their specialties, and the weekdays they work.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_availability",
            "description": "Find open appointment slots for one doctor in a date range. Only slots returned here can be booked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider_id": {"type": "string"},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD, defaults to tomorrow"},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD, defaults to 7 days after date_from"},
                    "part_of_day": {"type": "string", "enum": ["morning", "afternoon", "any"]},
                },
                "required": ["provider_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_appointments",
            "description": "List the verified caller's upcoming appointments.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an open slot for the verified caller.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id": {"type": "string"},
                    "reason": {"type": "string", "description": "Short visit reason, e.g. follow-up"},
                },
                "required": ["slot_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Move one of the verified caller's appointments to a new open slot. Atomic: the old time is released only if the new one is booked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "string"},
                    "new_slot_id": {"type": "string"},
                },
                "required": ["appointment_id", "new_slot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel one of the verified caller's appointments.",
            "parameters": {
                "type": "object",
                "properties": {"appointment_id": {"type": "string"}},
                "required": ["appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Hand the conversation to clinic staff. Use for emergencies, repeated failures, or anything outside scheduling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "urgency": {"type": "string", "enum": ["routine", "urgent", "emergency"]},
                },
                "required": ["reason", "urgency"],
            },
        },
    },
]

MUTATING_TOOLS = {"book_appointment", "reschedule_appointment", "cancel_appointment"}


@dataclass
class ToolEvent:
    name: str
    args: dict
    result: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "args": self.args, "result": self.result}


@dataclass
class ToolExecutor:
    clinic: Clinic
    faults: dict = field(default_factory=dict)
    patient_id: str | None = None
    failed_verifications: int = 0
    events: list = field(default_factory=list)

    def execute(self, name: str, raw_args: str | dict) -> dict:
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            if not isinstance(args, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError, TypeError):
            args = {}
            result = {"ok": False, "error": "invalid_arguments"}
            self.events.append(ToolEvent(name, {"raw": str(raw_args)[:200]}, result))
            return result
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            result = {"ok": False, "error": "unknown_tool"}
        elif self.faults.get(name) == "unavailable":
            result = {"ok": False, "error": "service_unavailable", "retryable": True}
        else:
            try:
                result = handler(**args)
            except TypeError:
                result = {"ok": False, "error": "invalid_arguments"}
        self.events.append(ToolEvent(name, args, result))
        return result

    def _require_patient(self) -> dict | None:
        if self.patient_id is None:
            return {"ok": False, "error": "not_verified", "hint": "Call verify_patient first."}
        return None

    def _tool_verify_patient(self, phone: str, date_of_birth: str) -> dict:
        if self.failed_verifications >= MAX_VERIFY_ATTEMPTS:
            return {"ok": False, "error": "verification_locked", "hint": "Escalate to staff."}
        digits = re.sub(r"\D", "", str(phone))[-10:]
        matches = [
            item for item in self.clinic.patients.values()
            if item.phone == digits and item.dob == str(date_of_birth).strip()
        ]
        if len(matches) != 1:
            # Same answer whether the phone or the date was wrong: no enumeration.
            self.failed_verifications += 1
            return {
                "ok": False,
                "error": "no_match",
                "attempts_left": MAX_VERIFY_ATTEMPTS - self.failed_verifications,
            }
        patient = matches[0]
        self.patient_id = patient.id
        self.failed_verifications = 0
        return {"ok": True, "patient": {"first_name": patient.first_name, "last_initial": patient.last_name[0]}}

    def _tool_list_providers(self) -> dict:
        return {
            "ok": True,
            "providers": [
                {"provider_id": item.id, "name": item.name, "specialty": item.specialty, "works_on": item.weekdays}
                for item in self.clinic.providers.values()
            ],
        }

    def _tool_search_availability(
        self,
        provider_id: str,
        date_from: str | None = None,
        date_to: str | None = None,
        part_of_day: str = "any",
    ) -> dict:
        if provider_id not in self.clinic.providers:
            return {"ok": False, "error": "unknown_provider"}
        today = self.clinic.today
        try:
            start = dt.date.fromisoformat(date_from) if date_from else today + dt.timedelta(days=1)
            end = dt.date.fromisoformat(date_to) if date_to else start + dt.timedelta(days=7)
        except ValueError:
            return {"ok": False, "error": "invalid_date", "hint": "Use YYYY-MM-DD."}
        start = max(start, today + dt.timedelta(days=1))
        end = min(end, today + dt.timedelta(days=self.clinic.window_days))
        slots = self.clinic.open_slots(provider_id, start, end)
        if part_of_day == "morning":
            slots = [item for item in slots if item[1] < "12:00"]
        elif part_of_day == "afternoon":
            slots = [item for item in slots if item[1] >= "12:00"]
        rows = [
            {
                "slot_id": slot_id(provider_id, day.isoformat(), time),
                "date": day.isoformat(),
                "weekday": WEEKDAYS[day.weekday()],
                "time": time,
            }
            for day, time in slots
        ]
        result = {"ok": True, "provider": self.clinic.providers[provider_id].name, "slots": rows[:MAX_SLOTS_RETURNED]}
        if len(rows) > MAX_SLOTS_RETURNED:
            result["more_available"] = len(rows) - MAX_SLOTS_RETURNED
        if not rows:
            result["note"] = "No open slots in this range."
        return result

    def _tool_get_my_appointments(self) -> dict:
        blocked = self._require_patient()
        if blocked:
            return blocked
        return {
            "ok": True,
            "appointments": [self._public(item) for item in self.clinic.active_for(self.patient_id)],
        }

    def _tool_book_appointment(self, slot_id: str, reason: str = "visit") -> dict:
        blocked = self._require_patient()
        if blocked:
            return blocked
        parsed = parse_slot_id(str(slot_id))
        if parsed is None:
            return {"ok": False, "error": "invalid_slot_id"}
        problem = self.clinic.is_bookable(*parsed)
        if problem:
            return {"ok": False, "error": problem}
        appointment = self.clinic.add_appointment(self.patient_id, *parsed, str(reason)[:80])
        return {"ok": True, "appointment": self._public(appointment)}

    def _tool_reschedule_appointment(self, appointment_id: str, new_slot_id: str) -> dict:
        blocked = self._require_patient()
        if blocked:
            return blocked
        current = self._owned(appointment_id)
        if current is None:
            return {"ok": False, "error": "appointment_not_found"}
        parsed = parse_slot_id(str(new_slot_id))
        if parsed is None:
            return {"ok": False, "error": "invalid_slot_id"}
        problem = self.clinic.is_bookable(*parsed)
        if problem:
            return {"ok": False, "error": problem, "note": "Original appointment unchanged."}
        moved = self.clinic.add_appointment(self.patient_id, *parsed, current.reason)
        current.status = "cancelled"
        return {"ok": True, "cancelled": self._public(current), "appointment": self._public(moved)}

    def _tool_cancel_appointment(self, appointment_id: str) -> dict:
        blocked = self._require_patient()
        if blocked:
            return blocked
        current = self._owned(appointment_id)
        if current is None:
            return {"ok": False, "error": "appointment_not_found"}
        current.status = "cancelled"
        return {"ok": True, "cancelled": self._public(current)}

    def _tool_escalate_to_human(self, reason: str, urgency: str = "routine") -> dict:
        if urgency not in {"routine", "urgent", "emergency"}:
            urgency = "routine"
        ticket = self.clinic.new_id("ticket")
        self.clinic.tickets.append(Ticket(ticket, str(reason)[:300], urgency, self.patient_id))
        return {"ok": True, "ticket_id": ticket, "urgency": urgency}

    def _owned(self, appointment_id: str):
        item = self.clinic.appointments.get(str(appointment_id))
        if item is None or item.patient_id != self.patient_id or item.status != "active":
            return None
        return item

    def _public(self, item) -> dict:
        provider = self.clinic.providers[item.provider_id]
        day = dt.date.fromisoformat(item.date)
        return {
            "appointment_id": item.id,
            "provider": provider.name,
            "date": item.date,
            "weekday": WEEKDAYS[day.weekday()],
            "time": item.time,
            "reason": item.reason,
        }
