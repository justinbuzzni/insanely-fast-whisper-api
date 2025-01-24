import os
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Body,
    BackgroundTasks,
    Request,
    UploadFile,
    File,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import pipeline
from .diarization_pipeline import diarize
import requests
import asyncio
import uuid
import shutil
from pathlib import Path
import tempfile
import subprocess




admin_key = os.environ.get(
    "ADMIN_KEY",
)

hf_token = os.environ.get(
    "HF_TOKEN",
)

# fly runtime env https://fly.io/docs/machines/runtime-environment
fly_machine_id = os.environ.get(
    "FLY_MACHINE_ID",
)
model_name = os.environ.get(
    "MODEL_NAME",
)
print("model_name:",model_name)
# model_name = "Systran/faster-whisper-large-v3"
is_use_faster = False
if "faster" in model_name:
    '''
    cudnn 8 을 설치해야 동작함 
    FROM nvcr.io/nvidia/pytorch:21.12-py3 
    이걸 써야함. 
    python 3.10 은 안됨. 
    그래서 해보려다가 공수가 너무 커져서 중단함
    '''
    from faster_whisper import WhisperModel
    # model_name == "Systran/faster-whisper-large-v3":
    is_use_faster = True
    # GPU 메모리 설정
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # WhisperModel 초기화
    model = WhisperModel(
        model_name,
        device="cuda",
        compute_type="float16",
    )
    pipe = None
else:
    # model="openai/whisper-large-v3-turbo",
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch.float16,
        device="cuda:0",
        model_kwargs=({"attn_implementation": "flash_attention_2"}),
    )
    model = None

# Replace the pipeline with WhisperModel



app = FastAPI()
loop = asyncio.get_event_loop()
running_tasks = {}


class WebhookBody(BaseModel):
    url: str
    header: dict[str, str] = {}


def pipeline_process(
    url: str,
    task: str,
    language: str,
    batch_size: int,
    timestamp: str,
    diarise_audio: bool,
    webhook: WebhookBody | None = None,
    task_id: str | None = None,
):
    errorMessage: str | None = None
    outputs = {}
    try:
        generate_kwargs = {
            "task": task,
            "language": None if language == "None" else language,
        }

        outputs = pipe(
            url,
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps="word" if timestamp == "word" else True,
        )

        if diarise_audio is True:
            speakers_transcript = diarize(
                hf_token,
                url,
                outputs,
            )
            outputs["speakers"] = speakers_transcript
    except asyncio.CancelledError:
        errorMessage = "Task Cancelled"
    except Exception as e:
        errorMessage = str(e)

    if task_id is not None:
        del running_tasks[task_id]

    if webhook is not None:
        webhookResp = (
            {"output": outputs, "status": "completed", "task_id": task_id}
            if errorMessage is None
            else {"error": errorMessage, "status": "error", "task_id": task_id}
        )

        if fly_machine_id is not None:
            webhookResp["fly_machine_id"] = fly_machine_id

        requests.post(
            webhook.url,
            headers=webhook.header,
            json=(webhookResp),
        )

    if errorMessage is not None:
        raise Exception(errorMessage)

    return outputs

def faster_process(
    url: str,
    task: str,
    language: str,
    batch_size: int,
    timestamp: str,
    diarise_audio: bool,
    webhook: WebhookBody | None = None,
    task_id: str | None = None,
):
    errorMessage: str | None = None
    outputs = {}
    try:
        # Convert language "None" to actual None
        language = None if language == "None" else language
        
        # Transcribe/translate with faster-whisper
        segments, info = model.transcribe(
            url,
            language=language,
            task=task,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True if timestamp == "word" else False,
        )

        # Format output to match original pipeline structure
        output_text = []
        chunks = []
        for segment in segments:
            chunk = {
                "text": segment.text,
                "timestamp": (segment.start, segment.end)
            }
            if timestamp == "word" and segment.words:
                chunk["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end
                    } for word in segment.words
                ]
            chunks.append(chunk)
            output_text.append(segment.text)

        outputs = {
            "text": "".join(output_text),
            "chunks": chunks,
            "language": info.language,
            "language_probability": info.language_probability
        }

        # Diarization 처리
        if diarise_audio:
            speakers_transcript = diarize(
                hf_token,
                url,
                outputs,
            )
            outputs["speakers"] = speakers_transcript

    except asyncio.CancelledError:
        errorMessage = "Task Cancelled"
    except Exception as e:
        errorMessage = str(e)

    # Task 관리
    if task_id is not None:
        del running_tasks[task_id]

    # Webhook 전송
    if webhook is not None:
        webhookResp = (
            {"output": outputs, "status": "completed", "task_id": task_id}
            if errorMessage is None
            else {"error": errorMessage, "status": "error", "task_id": task_id}
        )

        if fly_machine_id is not None:
            webhookResp["fly_machine_id"] = fly_machine_id

        requests.post(
            webhook.url,
            headers=webhook.header,
            json=(webhookResp),
        )

    if errorMessage is not None:
        raise Exception(errorMessage)

    return outputs

if is_use_faster:
    process = faster_process
else:
    process = pipeline_process
@app.middleware("http")
async def admin_key_auth_check(request: Request, call_next):
    if admin_key is not None:
        if ("x-admin-api-key" not in request.headers) or (
            request.headers["x-admin-api-key"] != admin_key
        ):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    response = await call_next(request)
    return response


@app.post("/")
def root(
    url: str = Body(),
    task: str = Body(default="transcribe", enum=["transcribe", "translate"]),
    language: str = Body(default="None"),
    batch_size: int = Body(default=64),
    timestamp: str = Body(default="chunk", enum=["chunk", "word"]),
    diarise_audio: bool = Body(
        default=False,
    ),
    webhook: WebhookBody | None = None,
    is_async: bool = Body(default=False),
    managed_task_id: str | None = Body(default=None),
):
    if url.lower().startswith("http") is False:
        raise HTTPException(status_code=400, detail="Invalid URL")

    if diarise_audio is True and hf_token is None:
        raise HTTPException(status_code=500, detail="Missing Hugging Face Token")

    if is_async is True and webhook is None:
        raise HTTPException(
            status_code=400, detail="Webhook is required for async tasks"
        )

    task_id = managed_task_id if managed_task_id is not None else str(uuid.uuid4())

    try:
        resp = {}
        if is_async is True:
            backgroundTask = asyncio.ensure_future(
                loop.run_in_executor(
                    None,
                    process,
                    url,
                    task,
                    language,
                    batch_size,
                    timestamp,
                    diarise_audio,
                    webhook,
                    task_id,
                )
            )
            running_tasks[task_id] = backgroundTask
            resp = {
                "detail": "Task is being processed in the background",
                "status": "processing",
                "task_id": task_id,
            }
        else:
            running_tasks[task_id] = None
            outputs = process(
                url,
                task,
                language,
                batch_size,
                timestamp,
                diarise_audio,
                webhook,
                task_id,
            )
            resp = {
                "output": outputs,
                "status": "completed",
                "task_id": task_id,
            }
        if fly_machine_id is not None:
            resp["fly_machine_id"] = fly_machine_id
        return resp
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def tasks():
    return {"tasks": list(running_tasks.keys())}


@app.get("/status/{task_id}")
def status(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]

    if task is None:
        return {"status": "processing"}
    elif task.done() is False:
        return {"status": "processing"}
    else:
        return {"status": "completed", "output": task.result()}


@app.delete("/cancel/{task_id}")
def cancel(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]
    if task is None:
        return HTTPException(status_code=400, detail="Not a background task")
    elif task.done() is False:
        task.cancel()
        del running_tasks[task_id]
        return {"status": "cancelled"}
    else:
        return {"status": "completed", "output": task.result()}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    task: str = Body(default="transcribe", enum=["transcribe", "translate"]),
    language: str = Body(default="None"),
    batch_size: int = Body(default=64),
    timestamp: str = Body(default="chunk", enum=["chunk", "word"]),
    diarise_audio: bool = Body(
        default=False,
    ),
    webhook: WebhookBody | None = None,
    is_async: bool = Body(default=False),
    managed_task_id: str | None = Body(default=None),
):
    # 허용된 파일 확장자 검사
    allowed_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    processed_file_path = temp_file_path
    converted_file = None

    try:
        # m4a 파일인 경우 wav로 변환
        if file_extension == '.m4a':
            converted_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            subprocess.run([
                'ffmpeg', '-i', temp_file_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                converted_file.name
            ], check=True)
            processed_file_path = converted_file.name

        task_id = managed_task_id if managed_task_id is not None else str(uuid.uuid4())
        
        if diarise_audio is True and hf_token is None:
            raise HTTPException(status_code=500, detail="Missing Hugging Face Token")

        if is_async is True and webhook is None:
            raise HTTPException(
                status_code=400, detail="Webhook is required for async tasks"
            )

        resp = {}
        if is_async is True:
            backgroundTask = asyncio.ensure_future(
                loop.run_in_executor(
                    None,
                    process,
                    processed_file_path,
                    task,
                    language,
                    batch_size,
                    timestamp,
                    diarise_audio,
                    webhook,
                    task_id,
                )
            )
            running_tasks[task_id] = backgroundTask
            resp = {
                "detail": "Task is being processed in the background",
                "status": "processing",
                "task_id": task_id,
            }
        else:
            running_tasks[task_id] = None
            outputs = process(
                processed_file_path,
                task,
                language,
                batch_size,
                timestamp,
                diarise_audio,
                webhook,
                task_id,
            )
            resp = {
                "output": outputs,
                "status": "completed",
                "task_id": task_id,
            }

        if fly_machine_id is not None:
            resp["fly_machine_id"] = fly_machine_id
        return resp

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        raise HTTPException(status_code=500, detail="Error converting audio file")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 임시 파일들 삭제
        try:
            os.unlink(temp_file_path)
            if converted_file:
                os.unlink(converted_file.name)
        except:
            pass
