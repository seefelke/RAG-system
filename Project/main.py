import Evaluation
from Testing import generate_groundtruth

if __name__ == "__main__":
    #folder = "logs/mistral-nemo FAISS/normal/" # Example path
    folder = "logs/GPT 4o Mini FAISS/normal/"
    evaluator = Evaluation.RAGEvaluator(folder, 0, 10)

    print("===== Metric Summary =====")

    summary = evaluator.summarize_metrics()
    for metric, stats in summary.items():
        print(f"\n--- {metric} ---")
        print(f"  Highest Average: {stats['highest_avg']:.4f} (File: {stats['highest_avg_file']})")
        print(f"  Lowest Average: {stats['lowest_avg']:.4f} (File: {stats['lowest_avg_file']})")
        print(f"  Average Across Runs: {stats['average']:.4f}")
        print(
            f"  Absolute Highest: {stats['absolute_highest']:.4f} (File: {stats['absolute_highest_file']}, Q: {stats['absolute_highest_question']})")
        print(
            f"  Absolute Lowest: {stats['absolute_lowest']:.4f} (File: {stats['absolute_lowest_file']}, Q: {stats['absolute_lowest_question']})")

    print("===== Best Overall Run =====")
    print(evaluator.best_overall_runs())

    print("===== Best Query Performance =====")
    print(evaluator.best_query_performance())

    print("===== Per-Question Averages =====")
    pq = evaluator.per_question_averages()
    print(pq)

    print("===== Per-Difficulty Averages =====")
    pq = evaluator.per_level_averages()
    print(pq)

    #generate_groundtruth(5, False)
