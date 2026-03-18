# Data Acquisition SOP — USGS Earthquake Catalog

## Data Source
- **API**: USGS ComCat (Comprehensive Earthquake Catalog)
- **URL**: `https://earthquake.usgs.gov/fdsnws/event/1/query`
- **Format**: CSV
- **Cost**: Free, no API key required

## Query Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `format` | `csv` | Returns CSV directly |
| `starttime` | `2000-01-01` | Start of date range |
| `endtime` | `2025-12-31` | End of date range |
| `minmagnitude` | `2.5` | Filter out micro-earthquakes |
| `orderby` | `time` | Chronological order |
| `limit` | `20000` | Max per request |

## Rate Limits & Pagination
- USGS limits responses to **20,000 rows** per query.
- The fetch script queries **year by year** to stay within limits.
- No explicit rate limit, but add a 1-second delay between requests to be respectful.

## CSV Columns (subset used)
`time`, `latitude`, `longitude`, `depth`, `mag`, `magType`, `place`, `type`

## Fallback Strategy
1. If the API is down, retry 3 times with exponential backoff.
2. If a particular year returns 0 rows, log a warning but continue.
3. If the total dataset is < 1,000 rows after all years, raise an error — something is wrong.

## Learnings
- (none yet — update as issues are encountered)
