from open_rubric.models.model import Model
from open_rubric.rubric import Rubric

if __name__ == "__main__":
    model = Model()
    rubric = Rubric.from_yaml("test_rubric.yaml")
    results = rubric.solve()
    breakpoint()
