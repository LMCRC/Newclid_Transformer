import os

need_aux = [
    "translated_imo_2012_p5",
    "translated_usamo_1997_p2",
]


if __name__ == "__main__":
    for problem in need_aux:
        os.system(
            f"alphageo --device cpu -o None --problems-file problems_datasets/llm_examplification.txt --problem {problem} --search-width 32 --search-depth 3 --batch-size 32 --lm-beam-width 32 --log-level 30"
        )
