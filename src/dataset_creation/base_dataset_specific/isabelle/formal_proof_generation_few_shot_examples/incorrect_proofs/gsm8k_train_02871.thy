theorem example:
    (* Gloria wants to buy the $129,000 mountain cabin that her friend Alfonso is selling. *)
    assumes "(Cabin_cost::nat) = 129000"
    (* She only has $150 in cash. *)
        and "(Gloria_cash::nat) = 150"
    (* She intends to raise the remaining amount by selling her mature trees for lumber. *)
    (* She has 20 cypress trees, 600 pine trees, and 24 maple trees. *)
        and "(Cypress_count::nat) = 20"
        and "(Pine_count::nat) = 600"
        and "(Maple_count::nat) = 24"
    (* She will get $100 for each cypress tree, $300 for a maple tree, and $200 per pine tree. *)
        and "(Cypress_price::nat) = 100"
        and "(Maple_price::nat) = 300"
        and "(Pine_price::nat) = 200"
    (* After paying Alfonso for the cabin, how much money will Gloria have left? *)
        and "(After_paying_Alfonso::nat) = Gloria_cash + Cypress_count * Cypress_price + Maple_count * Maple_price + Pine_count * Pine_price - Cabin_cost"
    (* Final Answer -- The answer is 200. *)
    shows "After_paying_Alfonso = 200"
proof -
    have "Cypress_count * Cypress_price = 2000"
        using assms by simp
    then have "Maple_count * Maple_price = 7200"
        using assms by simp
    then have "Pine_count * Pine_price = 120000"
        using assms by simp
    then have "Gloria_cash + Cypress_count * Cypress_price + Maple_count * Maple_price + Pine_count * Pine_price = 129200"
        using assms by simp
    then have "Gloria_cash + Cypress_count * Cypress_price + Maple_count * Maple_price + Pine_count * Pine_price - Cabin_cost = 200"
        using assms by simp
    then have "After_paying_Alfonso = 200"
        using assms by simp
    show ?thesis
        using assms by simp
qed
