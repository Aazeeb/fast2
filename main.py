
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Student API")

students = [
    {"id": 1, "name": "Rahul", "course": "Python"},
    {"id": 2, "name": "Priya", "course": "Data Science"},
    {"id": 3, "name": "Amit", "course": "AI"}
]


class Student(BaseModel):
    id: int
    name: str
    course: str


@app.get("/")
def home():
    return {"message": "Welcome to Student API"}


@app.get("/students")
def get_students():
    return students


@app.get("/students/{student_id}")
def get_student(student_id: int):
    for student in students:
        if student["id"] == student_id:
            return student

    raise HTTPException(
        status_code=404,
        detail="Student not found"
    )


@app.post("/students")
def add_student(student: Student):
    students.append(student.model_dump())

    return {
        "message": "Student added successfully",
        "student": student
    }
