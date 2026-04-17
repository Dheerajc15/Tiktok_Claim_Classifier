# Data

## Expected schema

| Column | Type | Description |
|---|---|---|
| `#` | int | Row identifier |
| `claim_status` | object | Target — `claim` or `opinion` |
| `video_id` | int | Unique video identifier |
| `video_duration_sec` | int | Video length in seconds (5–60) |
| `video_transcription_text` | object | Full transcription of the video |
| `verified_status` | object | `verified` or `not verified` |
| `author_ban_status` | object | `active`, `under review`, or `banned` |
| `video_view_count` | float | Number of views |
| `video_like_count` | float | Number of likes |
| `video_share_count` | float | Number of shares |
| `video_download_count` | float | Number of downloads |
| `video_comment_count` | float | Number of comments |

## Expected size

- ~19,382 rows
- 12 columns
- 298 rows contain null values across multiple fields
