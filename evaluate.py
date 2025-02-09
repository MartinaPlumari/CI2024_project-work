from src.utils.problemloader import ProblemList, Problem
import s310582 as sol

pl = ProblemList()
pl.load_from_path("data", 1, False, 10)

problem = pl.problems[0]

max_dist = 0
min_dist = 99999999
for i in range(problem.train_size):
	x = problem.x_train[:, i]
	y = problem.y_train[i]
	res = sol.f0(x)
	dist = abs(abs(y) - abs(res))
	if dist < min_dist:
		min_dist = dist
	if dist > max_dist:
		max_dist = dist
	print(f"actual_result: {y}\nfound_solution: {res}\ndistance: {dist}\n====================")

print(f"MIN DIST {min_dist}\nMAX DIST {max_dist}")

