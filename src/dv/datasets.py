"""Dataset registry shared by the analysis scripts. Civil (rule-based identity swaps, most rejected by the judges) is
kept in every computation but reported in the appendix only; MAIN is what the main text tables/figures show."""
ALL = ["cebab", "bios", "amzneg", "amzfp", "civil"]
MAIN = ["cebab", "bios", "amzneg", "amzfp"]
NAME = {"cebab": "CEBaB", "bios": "BiasBios", "amzneg": "Amazon-Neg", "amzfp": "Amazon-1P", "civil": "Civil"}
DESC = {"cebab": "service / food sentiment", "bios": "surgeon vs.\\ nurse / gender",
        "amzneg": "negative sentiment / negation", "amzfp": "negative sentiment / first person",
        "civil": "toxic / Muslim or LGBTQ"}
PANEL = {"cebab": "CEBaB: service | food", "bios": "BiasBios: surgeon | gender",
         "amzneg": "Amazon: sentiment | negation", "amzfp": "Amazon: sentiment | first person",
         "civil": "Civil: toxic | identity"}
# judge models per dataset (judge.parquet, judge_groq.parquet)
JUDGES = {ds: ("Qwen3-32B", "gpt-oss-120b") for ds in ALL}
JUDGES.update({"amzneg": ("Qwen3.8-27B", "gpt-oss-120b"), "amzfp": ("Qwen3.8-27B", "gpt-oss-120b")})
