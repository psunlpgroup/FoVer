theorem example:
    (* Janet hires six employees. *)
    (* Four of them are warehouse workers who make $15/hour, and the other two are managers who make $20/hour. *)
    assumes "(warehouse_workers::nat) = 4"
        and "(manager_workers::nat) = 2"
        and "(warehouse_wage::nat) = 15"
        and "(manager_wage::nat) = 20"
    (* Janet has to pay 10% of her workers' salaries in FICA taxes. *)
        and "(fica_rate::real) = 0.10"
    (* If everyone works 25 days a month and 8 hours a day, how much does Janet owe total for their wages and taxes for one month?. *)
        and "(work_days::nat) = 25"
        and "(work_hours_per_day::nat) = 8"
        and "(total_one_month::nat) = (warehouse_workers * work_days * work_hours_per_day * warehouse_wage + manager_workers * work_days * work_hours_per_day * manager_wage) + fica_rate * (warehouse_workers * work_days * work_hours_per_day * warehouse_wage + manager_workers * work_days * work_hours_per_day * manager_wage)"
    (* Final Answer -- The total amount Janet owes for wages and taxes is 7000 (salaries) + 700 (taxes) = 7700 dollars. *)
    shows "total_one_month = 7700"
proof -
    (* First, let's calculate the total hours worked per employee in a month. *)
    (* Each employee works 25 days × 8 hours/day = 200 hours. *)
    have "work_days * work_hours_per_day = 200"
        sledgehammer
    (* The warehouse workers make $15/hour, and there are 4 of them. *)
    (* Their total salary is 4 × 200 × 15 = 3000 dollars. *)
    then have "warehouse_workers * 200 * warehouse_wage = 3000"
        sledgehammer
    (* The managers make $20/hour, and there are 2 of them. *)
    (* Their total salary is 2 × 200 × 20 = 4000 dollars. *)
    then have "manager_workers * 200 * manager_wage = 4000"
        sledgehammer
    (* The total salary for all employees is 3000 + 4000 = 7000 dollars. *)
    then have "warehouse_workers * 200 * warehouse_wage + manager_workers * 200 * manager_wage = 7000"
        sledgehammer
    (* Janet has to pay 10% of the total salary in FICA taxes. *)
    (* 10% of 7000 is 0.10 × 7000 = 700 dollars. *)
    then have "fica_rate * (warehouse_workers * 200 * warehouse_wage + manager_workers * 200 * manager_wage) = 700"
        sledgehammer
    (* Therefore, the total amount Janet owes is 7000 + 700 = 7700 dollars. *)
    then have "total_one_month = warehouse_workers + fica_rate"
        sledgehammer
    then have "total_one_month = 7700"
        sledgehammer
    thus ?thesis
        sledgehammer
qed
