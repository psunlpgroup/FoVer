theorem example:
    (* Of the 3 friends, Harry has 4 times as many fish as Joe, and Joe has 8 times as many fish as Sam does. *)
    assumes "(Harry_fish::nat) = 4 * (Joe_fish::nat)"
        and "(Joe_fish::nat) = 8 * (Sam_fish::nat)"
    (* If Sam has 7 fish, how many fish does Harry have? *)
        and "(Sam_fish::nat) = 7"
    (* Final Answer -- The answer is 224. *)
    shows "Harry_fish = 224"
proof -
    have "Joe_fish = 8 * Sam_fish"
        using assms by simp
    then have "Joe_fish = 56"
        using assms by simp
    then have "Harry_fish = 4 * Joe_fish"
        using assms by simp
    then have "Harry_fish = 224"
        using assms by simp
    thus ?thesis
        using assms by simp
qed
