theorem example:
    (* It takes 1 hour for refrigerated dough to come to room temperature. *)
    assumes "(room_temp_time::real) = 1"
    (* It takes 15 minutes to shape the dough and 2 hours to proof. *)
        and "(shape_time::real) = 15 / 60"
        and "(proof_time::real) = 2"
    (* The bread takes 30 minutes to bake and 15 minutes to cool. *)
        and "(bake_time::real) = 30 / 60"
        and "(cool_time::real) = 15 / 60"
    (* If the bakery opens at 6:00 am, what is the latest time the head baker can make it to the store to start working?. *)
        and "(open_time::real) = 6"
        and "(latest_time::real) = (open_time - (room_temp_time + shape_time + proof_time + bake_time + cool_time))"
    (* Final Answer -- The head baker needs to arrive at the store by 2:00 am to start working. *)
    shows "latest_time = 2"
proof -
    have "room_temp_time + shape_time + proof_time + bake_time + cool_time = 1 + 15 / 60 + 2 + 30 / 60 + 15 / 60" 
        using assms by simp
    then have "room_temp_time + shape_time + proof_time + bake_time + cool_time = 4" 
        using assms by simp
    then have "latest_time = open_time - (room_temp_time + shape_time + proof_time + bake_time + cool_time)"
        using assms by simp
    then have "latest_time = 2"
        using assms by simp
    thus ?thesis
        using assms by simp
qed
