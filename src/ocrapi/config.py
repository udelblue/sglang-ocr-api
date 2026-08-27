from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    sglang_base_url: str = "http://localhost:30000/v1"
    sglang_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    sglang_api_key: str = "EMPTY"
    sglang_timeout_seconds: float = 120.0
    sglang_max_tokens: int = 4096

    max_upload_size_mb: int = 20
    allowed_content_types: tuple[str, ...] = (
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/webp",
    )
    pdf_rasterize_dpi: int = 200


settings = Settings()
