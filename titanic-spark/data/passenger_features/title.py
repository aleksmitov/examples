from typing import Any

from layer import Context, Dataset


def build_feature(context: Context, passengers: Dataset("titanic_dataset")) -> Any:
    passengers_df = passengers.to_spark()
    passengers_df.createOrReplaceTempView("passengers_df")
    # Get spark session from Layer Context.
    spark = context.spark()

    title_df = spark.sql("""
        WITH titleDF as (
            SELECT PassengerId,
                regexp_replace(
                    regexp_replace(
                        regexp_extract(Name, ' (\\\w+)\\\.',1)
                            ,'^(Don|Countess|Col|Rev|Lady|Capt|Dr|Sir|Jonkheer|Major)$', 'Rare')
                            ,'^(Mlle|Ms|Mme)$', 'Miss'
                    )
            as parsedTitle
        FROM passengers_df
        )

        SELECT PassengerId,
            CASE
                WHEN parsedTitle = "Mr" THEN 1
                WHEN parsedTitle = "Miss" THEN 2
                WHEN parsedTitle = "Mrs" THEN 3
                WHEN parsedTitle = "Master" THEN 4
                WHEN parsedTitle = "Rare" THEN 5
            END as TITLE
        FROM titleDF
    """)

    return title_df.select("PASSENGERID", "TITLE")
