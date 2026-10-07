"""In-memory scheduling backend loaded from data/clinic.json.

In production this would sit behind the clinic's scheduling API. The agent
never touches it directly; it goes through twocare.tools.
"""

from __future__ import annotations

import copy
import datetime as dt
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DATA_PATH = ROOT / "data" / "clinic.json"


@dataclass
class Patient:
    id: str
    first_name: str
    last_name: str
    phone: str
    dob: str


@dataclass
class Provider:
    id: str
    name: str
    specialty: str
    weekdays: list
    hours: list
    slot_minutes: int


@dataclass
class Appointment:
    id: str
    patient_id: str
    provider_id: str
    date: str
    time: str
    reason: str
    status: str = "active"  # active | cancelled


@dataclass
class Ticket:
    id: str
    reason: str
    urgency: str
    patient_id: str | None


def next_weekday(today: dt.date, weekday: str) -> dt.date:
    """The next occurrence of `weekday` strictly after today."""
    target = WEEKDAYS.index(weekday)
    delta = (target - today.weekday()) % 7 or 7
    return today + dt.timedelta(days=delta)


def slot_id(provider_id: str, date: str, time: str) -> str:
    return f"{provider_id}:{date}:{time.replace(':', '')}"


def parse_slot_id(value: str) -> tuple[str, str, str] | None:
    parts = value.split(":")
    if len(parts) != 3 or len(parts[2]) != 4:
        return None
    return parts[0], parts[1], f"{parts[2][:2]}:{parts[2][2:]}"


@dataclass
class Clinic:
    name: str
    today: dt.date
    window_days: int
    providers: dict
    patients: dict
    appointments: dict
    tickets: list = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1001))

    @classmethod
    def load(cls, today: dt.date, extra_appointments: list | None = None, path: Path = DATA_PATH) -> "Clinic":
        raw = json.loads(path.read_text())
        clinic = cls(
            name=raw["name"],
            today=today,
            window_days=raw["booking_window_days"],
            providers={item["id"]: Provider(**item) for item in raw["providers"]},
            patients={item["id"]: Patient(**item) for item in raw["patients"]},
            appointments={},
        )
        for item in raw["appointments"] + list(extra_appointments or []):
            date = next_weekday(today, item["weekday"]).isoformat()
            clinic.add_appointment(item["patient_id"], item["provider_id"], date, item["time"], item.get("reason", "visit"))
        return clinic

    def snapshot(self) -> dict:
        return {key: copy.copy(value) for key, value in self.appointments.items()}

    def new_id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._ids)}"

    def add_appointment(self, patient_id: str, provider_id: str, date: str, time: str, reason: str) -> Appointment:
        appointment = Appointment(self.new_id("apt"), patient_id, provider_id, date, time, reason)
        self.appointments[appointment.id] = appointment
        return appointment

    def is_bookable(self, provider_id: str, date: str, time: str) -> str | None:
        """Return None if the slot can be booked, else a reason code."""
        provider = self.providers.get(provider_id)
        if provider is None:
            return "unknown_provider"
        try:
            day = dt.date.fromisoformat(date)
        except ValueError:
            return "invalid_date"
        if day <= self.today:
            return "not_in_future"
        if day > self.today + dt.timedelta(days=self.window_days):
            return "outside_booking_window"
        if time not in self.provider_times(provider, day):
            return "not_a_clinic_time"
        if self.active_at(provider_id, date, time):
            return "slot_taken"
        return None

    def provider_times(self, provider: Provider, day: dt.date) -> list:
        if WEEKDAYS[day.weekday()] not in provider.weekdays:
            return []
        times = []
        for start, end in provider.hours:
            cursor = dt.datetime.combine(day, dt.time.fromisoformat(start))
            stop = dt.datetime.combine(day, dt.time.fromisoformat(end))
            while cursor < stop:
                times.append(cursor.strftime("%H:%M"))
                cursor += dt.timedelta(minutes=provider.slot_minutes)
        return times

    def active_at(self, provider_id: str, date: str, time: str) -> Appointment | None:
        for item in self.appointments.values():
            if item.status == "active" and (item.provider_id, item.date, item.time) == (provider_id, date, time):
                return item
        return None

    def open_slots(self, provider_id: str, date_from: dt.date, date_to: dt.date) -> list:
        provider = self.providers[provider_id]
        slots = []
        day = date_from
        while day <= date_to:
            for time in self.provider_times(provider, day):
                if self.is_bookable(provider_id, day.isoformat(), time) is None:
                    slots.append((day, time))
            day += dt.timedelta(days=1)
        return slots

    def active_for(self, patient_id: str) -> list:
        rows = [
            item for item in self.appointments.values()
            if item.patient_id == patient_id and item.status == "active"
        ]
        return sorted(rows, key=lambda item: (item.date, item.time))
