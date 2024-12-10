theorem example:
    (* Of the 3 friends, Harry has 4 times as many fish as Joe, and Joe has 8 times as many fish as Sam does. *)
    assumes "(Harry_fish::nat) = 4 * (Joe_fish::nat)"
        and "(Joe_fish::nat) = 8 * (Sam_fish::nat)"
    (* If Sam has 7 fish, how many fish does Harry have? *)
        and "(Sam_fish::nat) = 7"
    (* Final Answer -- The answer is 224. *)
    shows "Harry_fish = 224"
proof -
    (* Sam has 7 fish. *)
    (* Joe has 8 times as many fish as Sam, so Joe has 7 * 8 = 56 fish. *)
    have "Joe_fish = 7 * 8"
        sledgehammer
    then have "Joe_fish = 56"
        sledgehammer
    (* Harry has 4 times as many fish as Joe, so Harry has 56 * 4 = 224 fish. *)
    then have "Harry_fish = 56 * 4"
        sledgehammer
    (* Therefore, the answer (arabic numerals) is 224. *)
    then have "Harry_fish = 224"
        sledgehammer
    thus ?thesis
        sledgehammer
qed
