from open_rubric.rubric import Rubric
from open_rubric.models.model import Model
import time

if __name__ == "__main__":
    model = Model()
    rubric = Rubric.from_yaml("test_rubric.yaml")
    
    req = list(rubric.requirements.requirements.values())[2]
    dep_results = {"a": 1, "b": 2}
    req.evaluate(dep_results)
    print(f"{req.name}: {req._result.score} over {req._result.n_votes} votes")
    breakpoint()