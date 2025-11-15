"""
Database Schemas for PharmaSure

Each Pydantic model maps to a MongoDB collection (lowercased class name).
"""
from __future__ import annotations
from typing import Optional, List, Literal, Any
from datetime import datetime, date
from pydantic import BaseModel, Field, EmailStr

# Auth/account core
class Account(BaseModel):
    email: EmailStr
    password_hash: str
    role: Literal["user", "pharmacist"]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class User(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    address: Optional[str] = None
    dob: Optional[date] = None
    caregiver: bool = False

class Pharmacist(BaseModel):
    email: EmailStr
    pharmacy_name: str
    license_no: str
    address: Optional[str] = None
    contact: Optional[str] = None

# Catalog
class Medicine(BaseModel):
    name: str
    brand: Optional[str] = None
    active_ingredient: Optional[str] = None
    sku: Optional[str] = None
    barcode: Optional[str] = None
    otc_flag: bool = True

class Batch(BaseModel):
    medicine_id: str
    batch_no: str
    expiry_date: date
    quantity: int = Field(ge=0, default=0)
    price: float = Field(ge=0, default=0)

# User medicine list
class UserMedicine(BaseModel):
    user_id: str
    medicine_id: Optional[str] = None
    batch_id: Optional[str] = None
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    reminders: Optional[List[str]] = None  # ISO times or cron-like strings
    notes: Optional[str] = None
    expiry_date: Optional[date] = None

# Prescription
class Prescription(BaseModel):
    user_id: str
    pharmacist_id: Optional[str] = None
    file_url: str
    status: Literal["pending", "approved", "rejected"] = "pending"
    uploaded_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    notes: Optional[str] = None

# Order
class OrderItem(BaseModel):
    medicine_id: Optional[str] = None
    batch_id: Optional[str] = None
    name: Optional[str] = None
    qty: int
    price: float

class Order(BaseModel):
    user_id: str
    pharmacist_id: Optional[str] = None
    items: List[OrderItem]
    status: Literal["pending", "accepted", "ready", "dispensed", "rejected"] = "pending"
    total: float
    created_at: Optional[datetime] = None

# Chat
class ChatMessage(BaseModel):
    conversation_id: str
    from_role: Literal["user", "pharmacist"]
    to_role: Literal["user", "pharmacist"]
    content: str
    attachments: Optional[List[str]] = None
    timestamp: Optional[datetime] = None

# Notifications
class Notification(BaseModel):
    user_id: Optional[str] = None
    pharmacist_id: Optional[str] = None
    type: str
    payload: Any
    read_flag: bool = False
    created_at: Optional[datetime] = None
