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
    (* Convert all times into hours and sum them up. *)
    (* 1 hour (room temperature) + 0.25 hours (shaping) + 2 hours (proofing) + 0.5 hours (baking) + 0.25 hours (cooling) *)
    (* Total time required: 1 + 0.25 + 2 + 0.5 + 0.25 = 4 hours *)
    have "room_temp_time + shape_time + proof_time + bake_time + cool_time = 4" 
        sledgehammer
    (* The bakery opens at 6:00 am, so the baker must start working at least 4 hours before that. *)
    (* 6:00 am - 4 hours = 2:00 am *)
    (* Therefore, the head baker needs to arrive at the store by 2:00 am to start working. *)
    then have "open_time - 4 = 2"
        sledgehammer
    thus ?thesis
        sledgehammer
qed
