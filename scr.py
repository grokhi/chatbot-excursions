import json


def clean_prices(input_file: str = "excursions.json"):
    # Load JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Clean each excursion's prices
    for excursion in data:
        for price_group in excursion["prices"]:
            # Filter prices to keep only those with type
            price_group["prices"] = [price for price in price_group["prices"] if "type" in price]

    # Save cleaned data
    with open("_" + input_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    clean_prices()
