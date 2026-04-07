theorem example:
    (* Nancy and Jason are learning to dance for the first time. Nancy steps on her partner's feet 3 times as often as Jason. *)
    assumes "(Nancy_steps::nat) = 3 * (Jason_steps::nat)"
    (* If together they step on each other's feet 32 times, how many times does Jason step on his partner's feet? *)
        and "(Nancy_steps::nat) + (Jason_steps::nat) = 32"
    (* Final Answer -- The answer is 8. *)
    shows "(Jason_steps::nat) = 8"
proof -
    have "Nancy_steps = 3 * Jason_steps"
        using assms by simp
    then have "Nancy_steps + Jason_steps = 32"
        using assms by simp
    then have "3 * Jason_steps + Jason_steps = 32"
        using assms by simp
    then have "4 * Jason_steps = 32"
        using assms by simp
    then have "Jason_steps = 32 div 4"
        using assms by simp
    then have "Jason_steps = 8"
        using assms by simp
    thus ?thesis
        using assms by simp
qed
