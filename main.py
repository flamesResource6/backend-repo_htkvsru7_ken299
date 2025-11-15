import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Literal

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from passlib.context import CryptContext
from bson import ObjectId

from database import db, create_document
from schemas import (
    User as UserSchema, Pharmacist as PharmacistSchema,
    Medicine as MedicineSchema, Batch as BatchSchema, UserMedicine as UserMedicineSchema,
    Prescription as PrescriptionSchema, Order as OrderSchema, OrderItem as OrderItemSchema,
    ChatMessage as ChatMessageSchema
)

# FastAPI app
app = FastAPI(title="PharmaSure API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for uploaded prescriptions
STATIC_DIR = os.path.join(os.getcwd(), "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Auth setup
SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def oid(s: str):
    try:
        return ObjectId(s)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SignupBody(BaseModel):
    email: EmailStr
    password: str
    role: Literal["user", "pharmacist"]
    name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    dob: Optional[str] = None
    caregiver: Optional[bool] = False
    pharmacy_name: Optional[str] = None
    license_no: Optional[str] = None
    contact: Optional[str] = None


class LoginBody(BaseModel):
    email: EmailStr
    password: str


async def get_current_account(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        account_id: str = payload.get("sub")
        if account_id is None:
            raise credentials_exception
        role = payload.get("role")
        email = payload.get("email")
    except JWTError:
        raise credentials_exception
    acc = db["account"].find_one({"_id": ObjectId(account_id)})
    if not acc:
        raise credentials_exception
    return {"id": str(acc["_id"]), "email": email or acc.get("email"), "role": role or acc.get("role")}


@app.get("/")
def read_root():
    return {"app": "PharmaSure API", "status": "ok"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["connection_status"] = "Connected"
            collections = db.list_collection_names()
            response["collections"] = collections
            response["database"] = "✅ Connected & Working"
    except Exception as e:
        response["database"] = f"⚠️  Error: {str(e)[:80]}"
    return response


# Auth Endpoints
@app.post("/auth/signup", response_model=Token)
def signup(body: SignupBody):
    if db["account"].find_one({"email": body.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    account_doc = {
        "email": body.email,
        "password_hash": hash_password(body.password),
        "role": body.role,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    res = db["account"].insert_one(account_doc)
    account_id = str(res.inserted_id)

    if body.role == "user":
        profile = {
            "email": body.email,
            "name": body.name or "",
            "phone": body.phone,
            "address": body.address,
            "dob": body.dob,
            "caregiver": body.caregiver or False,
        }
        db["user"].insert_one(profile)
    else:
        profile = {
            "email": body.email,
            "pharmacy_name": body.pharmacy_name or "",
            "license_no": body.license_no or "",
            "address": body.address,
            "contact": body.contact,
        }
        db["pharmacist"].insert_one(profile)

    token = create_access_token({"sub": account_id, "role": body.role, "email": body.email})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/login", response_model=Token)
def login(body: LoginBody):
    acc = db["account"].find_one({"email": body.email})
    if not acc or not verify_password(body.password, acc.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    token = create_access_token({"sub": str(acc["_id"]), "role": acc["role"], "email": acc["email"]})
    return {"access_token": token, "token_type": "bearer"}


# Medicines & Batches
@app.get("/medicines")
def search_medicines(q: Optional[str] = None, skip: int = 0, limit: int = 20, _: dict = Depends(get_current_account)):
    filt = {}
    if q:
        filt = {"$or": [
            {"name": {"$regex": q, "$options": "i"}},
            {"brand": {"$regex": q, "$options": "i"}},
            {"active_ingredient": {"$regex": q, "$options": "i"}},
            {"barcode": {"$regex": q, "$options": "i"}},
        ]}
    items = list(db["medicine"].find(filt).skip(skip).limit(limit))
    for it in items:
        it["id"] = str(it.pop("_id"))
    return {"items": items}


class ScanPayload(BaseModel):
    barcode: str


@app.post("/medicines/scan")
def scan_medicine(payload: ScanPayload, _: dict = Depends(get_current_account)):
    med = db["medicine"].find_one({"barcode": payload.barcode})
    if not med:
        return {"matched": False, "medicine": None, "batches": []}
    med_id = str(med["_id"])
    batches = list(db["batch"].find({"medicine_id": med_id}))
    for b in batches:
        b["id"] = str(b.pop("_id"))
    med["id"] = med_id
    med.pop("_id", None)
    return {"matched": True, "medicine": med, "batches": batches}


# User medicines
@app.post("/users/{user_id}/medicines")
def add_user_medicine(user_id: str, body: UserMedicineSchema, me: dict = Depends(get_current_account)):
    if me["role"] != "user":
        raise HTTPException(status_code=403, detail="Only users can add medicines")
    data = body.model_dump()
    data["user_id"] = user_id
    ins_id = create_document("usermedicine", data)
    return {"id": ins_id}


@app.get("/users/{user_id}/medicines")
def list_user_medicines(user_id: str, expiry_threshold_days: int = 30, _: dict = Depends(get_current_account)):
    items = list(db["usermedicine"].find({"user_id": user_id}))
    warnings = []
    now = datetime.now(timezone.utc).date()
    for it in items:
        it["id"] = str(it.pop("_id"))
        exp = it.get("expiry_date")
        if isinstance(exp, str):
            try:
                exp = datetime.fromisoformat(exp).date()
            except Exception:
                exp = None
        if exp:
            days = (exp - now).days
            if days <= expiry_threshold_days:
                warnings.append({"user_medicine_id": it["id"], "name": it.get("name"), "days_to_expiry": days})
    return {"items": items, "warnings": warnings}


# Prescriptions
@app.post("/prescriptions")
def upload_prescription(
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    me: dict = Depends(get_current_account),
):
    if me["role"] != "user":
        raise HTTPException(status_code=403, detail="Only users can upload prescriptions")
    filename = f"{datetime.now(timezone.utc).timestamp()}_{file.filename}"
    dest = os.path.join(UPLOADS_DIR, filename)
    with open(dest, "wb") as f:
        f.write(file.file.read())
    file_url = f"/static/uploads/{filename}"
    pres = PrescriptionSchema(
        user_id=me["id"],
        file_url=file_url,
        status="pending",
        uploaded_at=datetime.now(timezone.utc),
        notes=notes,
    )
    pres_id = create_document("prescription", pres)
    return {"id": pres_id, "file_url": file_url}


@app.get("/pharma/prescriptions")
def pharmacist_list_prescriptions(me: dict = Depends(get_current_account)):
    if me["role"] != "pharmacist":
        raise HTTPException(status_code=403, detail="Pharmacist only")
    items = list(db["prescription"].find({"status": "pending"}).sort("uploaded_at", -1))
    for it in items:
        it["id"] = str(it.pop("_id"))
    return {"items": items}


@app.post("/prescriptions/{prescription_id}/approve")
def approve_prescription(prescription_id: str, me: dict = Depends(get_current_account)):
    if me["role"] != "pharmacist":
        raise HTTPException(status_code=403, detail="Pharmacist only")
    result = db["prescription"].update_one({"_id": oid(prescription_id)}, {"$set": {"status": "approved", "approved_at": datetime.now(timezone.utc), "pharmacist_id": me["id"]}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Prescription not found")
    p = db["prescription"].find_one({"_id": oid(prescription_id)})
    create_document("notification", {"user_id": p.get("user_id"), "type": "prescription_approved", "payload": {"prescription_id": prescription_id}, "read_flag": False, "created_at": datetime.now(timezone.utc)})
    return {"status": "approved"}


# Orders
class CreateOrderBody(BaseModel):
    items: List[OrderItemSchema]


@app.post("/orders")
def create_order(body: CreateOrderBody, me: dict = Depends(get_current_account)):
    if me["role"] != "user":
        raise HTTPException(status_code=403, detail="Only users can create orders")
    total = sum([(it.qty * it.price) for it in body.items])
    order = OrderSchema(user_id=me["id"], items=body.items, status="pending", total=total, created_at=datetime.now(timezone.utc))
    order_id = create_document("order", order)
    return {"id": order_id, "status": "pending", "total": total}


@app.get("/pharma/orders")
def pharmacist_orders(me: dict = Depends(get_current_account)):
    if me["role"] != "pharmacist":
        raise HTTPException(status_code=403, detail="Pharmacist only")
    items = list(db["order"].find({}).sort("created_at", -1))
    for it in items:
        it["id"] = str(it.pop("_id"))
    return {"items": items}


class OrderStatusBody(BaseModel):
    status: Literal["accepted", "ready", "dispensed", "rejected"]


@app.patch("/orders/{order_id}/status")
def update_order_status(order_id: str, body: OrderStatusBody, me: dict = Depends(get_current_account)):
    if me["role"] != "pharmacist":
        raise HTTPException(status_code=403, detail="Pharmacist only")
    result = db["order"].update_one({"_id": oid(order_id)}, {"$set": {"status": body.status}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    ord_doc = db["order"].find_one({"_id": oid(order_id)})
    create_document("notification", {"user_id": ord_doc.get("user_id"), "type": "order_status", "payload": {"order_id": order_id, "status": body.status}, "read_flag": False, "created_at": datetime.now(timezone.utc)})
    return {"status": body.status}


# Chat
class ChatBody(BaseModel):
    conversation_id: str
    to_role: Literal["user", "pharmacist"]
    content: str
    attachments: Optional[List[str]] = None


@app.post("/chats")
def post_chat(body: ChatBody, me: dict = Depends(get_current_account)):
    msg = ChatMessageSchema(
        conversation_id=body.conversation_id,
        from_role=me["role"],
        to_role=body.to_role,
        content=body.content,
        attachments=body.attachments,
        timestamp=datetime.now(timezone.utc)
    )
    mid = create_document("chatmessage", msg)
    return {"id": mid}


@app.get("/chats/{conversation_id}")
def get_chat(conversation_id: str, _: dict = Depends(get_current_account)):
    msgs = list(db["chatmessage"].find({"conversation_id": conversation_id}).sort("timestamp", 1))
    for m in msgs:
        m["id"] = str(m.pop("_id"))
    return {"messages": msgs}


# Notifications
@app.get("/notifications")
def list_notifications(me: dict = Depends(get_current_account)):
    filt = {"user_id": me["id"]} if me["role"] == "user" else {"pharmacist_id": me["id"]}
    items = list(db["notification"].find(filt).sort("created_at", -1))
    for it in items:
        it["id"] = str(it.pop("_id"))
    return {"items": items}


@app.patch("/notifications/{notification_id}/read")
def mark_notification_read(notification_id: str, me: dict = Depends(get_current_account)):
    result = db["notification"].update_one({"_id": oid(notification_id)}, {"$set": {"read_flag": True}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "ok"}


# Analytics (basic)
@app.get("/analytics/pharma")
def analytics_pharma(me: dict = Depends(get_current_account)):
    if me["role"] != "pharmacist":
        raise HTTPException(status_code=403, detail="Pharmacist only")
    low_stock = db["batch"].count_documents({"quantity": {"$lt": 5}})
    now = datetime.now(timezone.utc)
    soon = now + timedelta(days=30)
    expiring = db["batch"].count_documents({"expiry_date": {"$lte": soon}})
    orders_count = db["order"].count_documents({})
    return {"low_stock": low_stock, "expiring_batches": expiring, "orders": orders_count}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
