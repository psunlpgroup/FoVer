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
    (* Gloria needs 129,000 dollars for the cabin, but she only has 150 dollars. *)
    (* So she needs 129,000 - 150 = 128,850 dollars. *)
    have "Cabin_cost - Gloria_cash = 128850"
        sledgehammer
    (* She will get 100 dollars for each of the 20 cypress trees, which is 20 × 100 = 2,000 dollars. *)
    then have "Cypress_count * Cypress_price = 2000"
        sledgehammer
    (* She will get 300 dollars for each of the 24 maple trees, which is 24 × 300 = 7,200 dollars. *)
    then have "Maple_count * Maple_price = 7200"
        sledgehammer
    (* She will get 200 dollars for each of the 600 pine trees, which is 600 × 200 = 120,000 dollars. *)
    then have "Pine_count * Pine_price = 120000"
        sledgehammer
    (* In total, she will get 2,000 + 7,200 + 120,000 = 129,200 dollars from selling the trees. *)
    then have "Cypress_count * Cypress_price + Maple_count * Maple_price + Pine_count * Pine_price = 129200"
        sledgehammer
    (* After selling the trees, she will have 129,200 dollars. *)
    (* She will pay 129,000 dollars for the cabin, so she will have 129,200 - 129,000 = 200 dollars left. *)
    then have "(Cypress_count * Cypress_price + Maple_count * Maple_price + Pine_count * Pine_price) - Cabin_cost = 200"
        sledgehammer
    (* Therefore, the answer (arabic numerals) is 200. *)
    then have "After_paying_Alfonso = 200"
        sledgehammer
    show ?thesis
        sledgehammer
qed
