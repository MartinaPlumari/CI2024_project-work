from ProblemLoader import Problem

class Saver:
    base_path : str
    student_name : str
    student_id : str
    template : str

    def __init__(self, path : str, student_name : str, student_id : str):
        self.base_path = path
        self.student_name = student_name
        self.student_id = student_id
        self.template = f"""# CI 2024 final project\n# Student {self.student_name}\n# ID {self.student_id}\n\nimport numpy as np\n\n"""
        self.generate_solution_file(f"{self.base_path}{self.student_id}.py")

    def generate_solution_file(self, path : str) -> None:
        with open(path , '+w') as f:
            f.seek(0)
            f.write(self.template)
            f.close()

    def append_solution(self, problem : Problem) -> None:
        assert problem.solution == "", "The solution is empty! Please try using a proper solution."

        solution_function = f"def f{problem.problem_id}(n : np.ndarray) -> np.ndarray:\n\treturn {problem.solution}\n\n"

        with open(f"{self.base_path}{self.student_id}.py", 'a') as f:
            f.write(solution_function)
            f.close()

