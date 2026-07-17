"""Generate sample US macro economic stress test Excel file."""

import pandas as pd

from src.config.paths import DATA_DIR, MACRO_XLSX_FILE, ensure_directory

SCENARIOS = {
    "Normal": [
        ("unemployment_rate", 3.8, 3.8, 1.00),
        ("gdp_growth", 2.1, 2.1, 1.00),
        ("interest_rate", 5.25, 5.25, 1.00),
        ("hpi_change", 3.5, 3.5, 1.00),
    ],
    "Boom": [
        ("unemployment_rate", 3.8, 3.2, 0.85),
        ("gdp_growth", 2.1, 3.5, 0.85),
        ("interest_rate", 5.25, 4.75, 0.90),
        ("hpi_change", 3.5, 6.0, 0.85),
    ],
    "Recession": [
        ("unemployment_rate", 3.8, 7.5, 1.35),
        ("gdp_growth", 2.1, -1.5, 1.40),
        ("interest_rate", 5.25, 6.50, 1.15),
        ("hpi_change", 3.5, -8.0, 1.30),
    ],
}


def main() -> None:
    ensure_directory(DATA_DIR)
    output = MACRO_XLSX_FILE

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, rows in SCENARIOS.items():
            df = pd.DataFrame(
                rows,
                columns=["variable", "base_value", "stressed_value", "pd_multiplier"],
            )
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Created {output}")


if __name__ == "__main__":
    main()
