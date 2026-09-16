import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from query import answer_question
from upload_service import process_pdf
from upload_service import create_supabase_client
from typing import Optional


app = FastAPI()


# =========================
# Request Models
# =========================

class QueryRequest(BaseModel):
    question: str
    document_id: Optional[int] = None


class FeedbackRequest(BaseModel):
    log_id: int
    feedback: str


# =========================
# Home
# =========================

@app.get("/")
def home():
    return {
        "message": "Enterprise Knowledge Assistant API is running"
    }


# =========================
# Query
# =========================

@app.post("/query")
def query(request: QueryRequest):

    start_time = time.time()

    try:
        # 1. 调用 RAG
        result = answer_question(
            request.question,
            document_id=request.document_id
        )

        # 2. 计算响应时间
        latency = round(
            time.time() - start_time,
            3
        )

        answer = result.get(
            "answer",
            ""
        )

        sources = result.get(
            "sources",
            []
        )

        # 3. Supabase
        supabase = create_supabase_client()

        # 4. 写 query log
        log_response = (
            supabase
            .table("query_logs")
            .insert({
                "question":
                    request.question,

                "answer":
                    answer,

                "sources":
                    sources,

                "latency":
                    latency,

                "feedback":
                    None
            })
            .execute()
        )

        # 5. 获取 log id
        log_id = None

        if log_response.data:
            log_id = (
                log_response
                .data[0]["id"]
            )

        # 6. 返回
        return {
            "answer":
                answer,

            "sources":
                sources,

            "latency":
                latency,

            "log_id":
                log_id
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================
# Feedback
# =========================

@app.post("/feedback")
def submit_feedback(
    request: FeedbackRequest
):

    # 只允许这两个值
    if request.feedback not in [
        "helpful",
        "not_helpful"
    ]:
        raise HTTPException(
            status_code=400,
            detail=(
                "feedback must be "
                "'helpful' or 'not_helpful'"
            )
        )

    try:
        supabase = create_supabase_client()

        response = (
            supabase
            .table("query_logs")
            .update({
                "feedback":
                    request.feedback
            })
            .eq(
                "id",
                request.log_id
            )
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Log not found."
            )

        return {
            "message":
                "Feedback saved successfully.",

            "log_id":
                request.log_id,

            "feedback":
                request.feedback
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================
# Upload PDF
# =========================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    if (
        file.content_type
        != "application/pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:
        file_bytes = await file.read()

        result = process_pdf(
            file_bytes=file_bytes,
            filename=file.filename
        )

        return {
            "message":
                "Document uploaded successfully.",

            **result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
    
@app.get("/analytics")
def analytics():
    try:
        supabase = create_supabase_client()

        response = (
            supabase
            .table("query_logs")
            .select("*")
            .order(
                "created_at",
                desc=True
            )
            .execute()
        )

        logs = response.data or []

        total_questions = len(logs)

        latencies = [
            log["latency"]
            for log in logs
            if log.get("latency") is not None
        ]

        average_latency = (
            round(
                sum(latencies) / len(latencies),
                3
            )
            if latencies
            else 0
        )

        feedback_logs = [
            log
            for log in logs
            if log.get("feedback")
            in [
                "helpful",
                "not_helpful"
            ]
        ]

        helpful_count = sum(
            1
            for log in feedback_logs
            if log.get("feedback")
            == "helpful"
        )

        feedback_count = len(
            feedback_logs
        )

        helpful_rate = (
            round(
                helpful_count
                / feedback_count
                * 100,
                1
            )
            if feedback_count
            else 0
        )

        not_helpful_questions = [
            {
                "id":
                    log.get("id"),

                "question":
                    log.get("question"),

                "answer":
                    log.get("answer"),

                "latency":
                    log.get("latency"),

                "created_at":
                    log.get("created_at")
            }
            for log in logs
            if log.get("feedback")
            == "not_helpful"
        ]

        recent_logs = [
            {
                "id":
                    log.get("id"),

                "question":
                    log.get("question"),

                "latency":
                    log.get("latency"),

                "feedback":
                    log.get("feedback"),

                "created_at":
                    log.get("created_at")
            }
            for log in logs[:20]
        ]

        return {
            "total_questions":
                total_questions,

            "average_latency":
                average_latency,

            "helpful_rate":
                helpful_rate,

            "feedback_count":
                feedback_count,

            "not_helpful_questions":
                not_helpful_questions,

            "recent_logs":
                recent_logs
        }
    except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=str(error)
            )
    
@app.get("/documents")
def get_documents():
    try:
        supabase = create_supabase_client()

        response = (
            supabase
            .table("documents")
            .select("id, filename, created_at")
            .order(
                "created_at",
                desc=True
            )
            .execute()
        )

        return {
            "documents":
                response.data or []
        }
    
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )