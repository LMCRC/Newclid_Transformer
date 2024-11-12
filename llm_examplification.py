import os

need_aux = [
    "translated_imo_2008_p1_all_extras_center",
]


if __name__ == "__main__":
    for problem in need_aux:
        os.system(
            f"alphageo --device cpu -o exp --problems-file problems_datasets/llm_examplification.txt --problem {problem} --search-width 32 --search-depth 3 --batch-size 32 --lm-beam-width 32 --log-level 20"
        )
