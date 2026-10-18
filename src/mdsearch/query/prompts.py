PARSE_SYSTEM = """You turn a shopper's free-text product search query into structured search parameters.

Return a JSON object with:
- "semantic_text": what the product is and the qualities the shopper wants \
(e.g. "creative building set", "suitable for formal events"). Remove everything you put into \
filters. If nothing is left, use the product-type words from the query. Never empty.
- "filters": ONLY constraints the shopper stated explicitly. Use null for anything not stated; \
never guess.
  - "brand": the brand/manufacturer as written.
  - "price_min" / "price_max": numbers only, ignore currency symbols. "under 40" -> price_max 40. \
"over 20" -> price_min 20. "around 12" -> price_min 9.6, price_max 14.4 (plus/minus 20%).
  - "rating_min": minimum star rating. "top-rated" or "highly rated" -> 4.0 unless a number is given.
  - "age_min" / "age_max": the child's age in years. "for a 3-year-old" -> both 3. \
"aged 5 to 8" -> age_min 5, age_max 8."""

PARSE_USER = """Examples:

Query: Looking for educational toys specifically from LEGO, designed to promote creativity, suitable for children aged 5-8
{{"semantic_text": "educational toys that promote creativity", "filters": {{"brand": "LEGO", "price_min": null, "price_max": null, "rating_min": null, "age_min": 5, "age_max": 8}}}}

Query: Locate a top-rated board game from Hasbro for kids aged 9-12 within a budget of 40
{{"semantic_text": "board game", "filters": {{"brand": "Hasbro", "price_min": null, "price_max": 40, "rating_min": 4.0, "age_min": 9, "age_max": 12}}}}

Query: fun puzzle to keep a toddler busy on long car rides
{{"semantic_text": "fun puzzle to keep a toddler busy on long car rides", "filters": {{"brand": null, "price_min": null, "price_max": null, "rating_min": null, "age_min": null, "age_max": null}}}}

Now the real query.

Query: {query}"""


def build_parse_prompt(query: str) -> str:
    return PARSE_USER.format(query=query)
