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
    have "work_days * work_hours_per_day = 200"
        using assms by simp
    then have "warehouse_workers * work_days * work_hours_per_day * warehouse_wage = 3000"
        using assms by simp
    then have "manager_workers * work_days * work_hours_per_day * manager_wage = 4000"
        using assms by simp
    then have "warehouse_workers * work_days * work_hours_per_day * warehouse_wage + manager_workers * work_days * work_hours_per_day * manager_wage = 7000"
        using assms by simp
    then have "fica_rate * (warehouse_workers * work_days * work_hours_per_day * warehouse_wage + manager_workers * work_days * work_hours_per_day * manager_wage) = 700"
        using assms by simp
    then have "total_one_month = warehouse_workers + fica_rate"
        using assms by simp
    then have "total_one_month = 7700"
        using assms by simp
    thus ?thesis
        using assms by simp
qed
