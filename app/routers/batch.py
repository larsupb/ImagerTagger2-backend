import asyncio
import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from app.sessions import Session, get_session
from app.models.schemas import (
    BatchProcessRequest,
    BucketAnalyzeRequest,
    BucketAnalyzeResponse,
    ColorMatchPreviewRequest,
    ColorMatchPreviewResult,
    WhiteBalancePreviewRequest,
    WhiteBalancePreviewResult,
)
from app.config import read_settings
from app.services.caption_service import save_caption as db_save_caption, read_caption as db_read_caption
from app.services.version_service import create_version_backup
from app.services import task_service

router = APIRouter(prefix="/api/batch", tags=["batch"])


@router.post("/process")
async def batch_process(
    req: BatchProcessRequest, session: Session = Depends(get_session)
):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")

    ds = session.dataset
    settings = read_settings()

    buckets = None
    bucket_map = None
    if req.bucket_resize:
        from lib.bucketing import generate_buckets, analyze_buckets

        buckets = generate_buckets(
            req.bucket_resolution, req.bucket_step, req.bucket_max_steps
        )
        assignment = analyze_buckets(ds, buckets)
        bucket_map = {}
        for bucket, indices in assignment.items():
            for idx in indices:
                bucket_map[idx] = bucket

    total = len(ds)
    task_id = task_service.create_task(total)
    task_service.start_task(task_id)

    async def run_batch():
        try:
            for i in range(total):
                if task_service.is_cancelled(task_id):
                    break

                item = ds.get_item(i)
                log_line = []

                try:
                    if req.rename:
                        offset = getattr(req, "rename_offset", 0) or 0
                        new_name = f"{i + offset:05d}"
                        ds.rename_item(i, new_name)
                        log_line.append(f"Renamed to {new_name}")

                    if req.upscale:
                        create_version_backup(session, i, "upscale")
                        from PIL import Image

                        upscaler_name = getattr(req, "upscaler", None) or settings.get(
                            "upscaler", "NMKD_Siax_200k_4x"
                        )
                        target_mp = settings.get("upscale_target_megapixels", 2.0)

                        if upscaler_name.startswith("comfyui-"):
                            from app.services import processing_service

                            await asyncio.to_thread(
                                processing_service.upscale_image,
                                session,
                                i,
                                upscaler_name,
                                target_mp,
                            )
                        else:
                            from lib.upscaling import upscale_image as do_upscale

                            img = Image.open(item.media_path)
                            result = await asyncio.to_thread(
                                do_upscale,
                                img,
                                upscaler_name,
                                session.config,
                                target_megapixels=target_mp,
                            )
                            result.save(item.media_path)
                        log_line.append("Upscaled")

                    if req.bucket_resize and bucket_map and i in bucket_map:
                        create_version_backup(session, i, "bucket_resize")
                        from lib.bucketing import resize_and_crop_to_bucket
                        from PIL import Image

                        target_bucket = bucket_map[i]
                        img = Image.open(item.media_path)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        if (
                            img.width < target_bucket.width
                            or img.height < target_bucket.height
                        ):
                            from lib.upscaling import upscale_image as do_upscale

                            img = do_upscale(
                                img,
                                settings.get("upscaler", "NMKD_Siax_200k_4x"),
                                session.config,
                                max_current_megapixels=999,
                                target_megapixels=999,
                            )
                        resized = resize_and_crop_to_bucket(img, target_bucket)
                        resized.save(item.media_path)
                        log_line.append(
                            f"Resized to bucket {target_bucket.width}x{target_bucket.height}"
                        )

                    if req.mask:
                        from lib.masking import ask_mask_from_model
                        from PIL import Image

                        img = Image.open(item.media_path)
                        mask = await asyncio.to_thread(
                            ask_mask_from_model, img, "u2net_human_seg"
                        )
                        mask_path = item.mask_path
                        if mask_path:
                            import os

                            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                            mask.save(mask_path)
                        log_line.append("Mask generated")

                    if req.caption:
                        skip = req.skip_existing_captions and bool(db_read_caption(session, i, req.caption_type or "tags"))
                        if skip:
                            log_line.append("Caption skipped (existing)")
                        else:
                            tagger_name = req.tagger
                            if tagger_name == "unified":
                                caption = getattr(req, "unified_caption", "") or ""
                            elif tagger_name == "vlm-tagger" and req.vlm_prompt:
                                from lib.tagging.openai_tagger import generate_vlm_caption
                                from app.config import vlm_endpoint_settings

                                vlm_settings = {**vlm_endpoint_settings(), "prompt": req.vlm_prompt}
                                caption = await asyncio.to_thread(
                                    generate_vlm_caption,
                                    item.media_path,
                                    vlm_settings,
                                )
                            else:
                                from lib.captioning import generate_caption as gen_caption

                                caption = await asyncio.to_thread(
                                    gen_caption,
                                    i,
                                    tagger_name,
                                    {"dataset": session.dataset},
                                )
                            db_save_caption(session, i, caption, req.caption_type or "tags")
                            log_line.append(f"Caption: {caption[:50]}...")

                    if req.white_balance:
                        from lib.white_balance import white_balance_image
                        from PIL import Image

                        try:
                            img = Image.open(item.media_path)
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            create_version_backup(session, i, "white_balance")
                            result = await asyncio.to_thread(
                                white_balance_image,
                                img,
                                req.white_balance_method or "gray_world",
                            )
                            result.save(item.media_path)
                            log_line.append(
                                f"White balanced using {req.white_balance_method or 'gray_world'}"
                            )
                        except Exception as e:
                            log_line.append(f"White balance ERROR: {str(e)}")

                    if req.color_match:
                        from lib.color_matching import color_match_image
                        from PIL import Image

                        ref_index = req.color_match_reference
                        if ref_index < 0 or ref_index >= len(ds):
                            log_line.append(
                                "Color match: SKIPPED - invalid reference index"
                            )
                        else:
                            ref_item = ds.get_item(ref_index)
                            try:
                                ref_img = Image.open(ref_item.media_path)
                                create_version_backup(session, i, "color_match")

                                img = Image.open(item.media_path)
                                result = await asyncio.to_thread(
                                    color_match_image,
                                    img,
                                    ref_img,
                                    req.color_match_method or "histogram",
                                )
                                result.save(item.media_path)
                                log_line.append(
                                    f"Color matched using {req.color_match_method or 'histogram'}"
                                )
                            except Exception as e:
                                log_line.append(f"Color match ERROR: {str(e)}")

                except Exception as e:
                    log_line.append(f"ERROR: {str(e)}")

                await asyncio.sleep(0)
                task_service.update_progress(
                    task_id,
                    i,
                    total,
                    item.filename,
                    "; ".join(log_line) if log_line else "Processed",
                )
                await asyncio.sleep(0)

            task_service.complete_task(task_id)
        except Exception as e:
            task_service.complete_task(task_id, str(e))

    asyncio.create_task(run_batch())
    return {"task_id": task_id}


@router.post("/analyze-buckets", response_model=BucketAnalyzeResponse)
def analyze_buckets(req: BucketAnalyzeRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")

    from lib.bucketing import generate_buckets, analyze_buckets

    ds = session.dataset
    buckets = generate_buckets(req.resolution, req.step, req.max_steps)
    result = analyze_buckets(ds, buckets)

    bucket_list = []
    for bucket, indices in result.items():
        bucket_list.append(
            {
                "width": bucket.width,
                "height": bucket.height,
                "count": len(indices),
                "images": [ds.get_item(i).filename for i in indices[:10]],
            }
        )

    return BucketAnalyzeResponse(
        buckets=bucket_list,
        total_images=sum(len(v) for v in result.values()),
    )


@router.post("/preview-color-match", response_model=ColorMatchPreviewResult)
async def preview_color_match(
    req: ColorMatchPreviewRequest, session: Session = Depends(get_session)
):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")

    ds = session.dataset
    total = len(ds)

    if req.reference < 0 or req.reference >= total:
        raise HTTPException(400, "Invalid reference index")

    import random
    import base64
    import io

    indices = random.sample(range(total), min(req.sample_count, total))
    ref_item = ds.get_item(req.reference)

    from lib.color_matching import color_match_image
    from PIL import Image

    ref_img = Image.open(ref_item.media_path)
    if ref_img.mode != "RGB":
        ref_img = ref_img.convert("RGB")

    previews = []
    for idx in indices:
        item = ds.get_item(idx)
        source_img = Image.open(item.media_path)
        if source_img.mode != "RGB":
            source_img = source_img.convert("RGB")

        matched_img = color_match_image(source_img, ref_img, req.method)

        source_buf = io.BytesIO()
        matched_buf = io.BytesIO()
        source_img.save(source_buf, format="JPEG", quality=85)
        matched_img.save(matched_buf, format="JPEG", quality=85)

        previews.append(
            {
                "index": idx,
                "filename": item.filename,
                "before": base64.b64encode(source_buf.getvalue()).decode(),
                "after": base64.b64encode(matched_buf.getvalue()).decode(),
            }
        )

    return ColorMatchPreviewResult(previews=previews)


@router.post("/preview-white-balance", response_model=WhiteBalancePreviewResult)
async def preview_white_balance(
    req: WhiteBalancePreviewRequest, session: Session = Depends(get_session)
):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")

    ds = session.dataset
    total = len(ds)

    import random
    import base64
    import io

    indices = random.sample(range(total), min(req.sample_count, total))

    from lib.white_balance import white_balance_image
    from PIL import Image

    previews = []
    for idx in indices:
        item = ds.get_item(idx)
        source_img = Image.open(item.media_path)
        if source_img.mode != "RGB":
            source_img = source_img.convert("RGB")

        result_img = white_balance_image(source_img, req.method)

        source_buf = io.BytesIO()
        result_buf = io.BytesIO()
        source_img.save(source_buf, format="JPEG", quality=85)
        result_img.save(result_buf, format="JPEG", quality=85)

        previews.append(
            {
                "index": idx,
                "filename": item.filename,
                "before": base64.b64encode(source_buf.getvalue()).decode(),
                "after": base64.b64encode(result_buf.getvalue()).decode(),
            }
        )

    return WhiteBalancePreviewResult(previews=previews)
