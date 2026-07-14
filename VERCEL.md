# Vercel deployment

The playground uses a private Vercel Blob store for durable annotation data.
Each authorized annotation is written as an immutable JSON object under
`annotations/v1/`. The deployment deliberately returns `503` instead of
writing to ephemeral `/tmp` when durable storage is unavailable.

Production inference uses the committed ONNX exports beside the playground
checkpoint. Regenerate them after changing the checkpoint with:

```bash
python scripts/export_playground_onnx.py
```

Required Vercel environment variables:

- `BLOB_READ_WRITE_TOKEN`: supplied by a private Vercel Blob store connected
  to the project.
- `SLM_ANNOTATION_TOKEN`: a private bearer token shared with authorized
  annotators. Enter it in the playground's Advanced section.

Both values must be configured for Production, Preview, and Development when
those environments should accept annotations. Never expose either value with
a `NEXT_PUBLIC_` or `VITE_` prefix.
