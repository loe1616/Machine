"""The template of the main script of the machine learning process
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameStatus, PlatformAction
)

def ml_loop():
    """The main loop of the machine learning process

    This loop is run in a separate process, and communicates with the game process.

    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """
    import pickle
    import numpy as np
	
	
    vx = 0
    vy = 0
    filename = "C:\\Users\\user\\Downloads\\MLGame-master\\games\\arkanoid\\ml\\knn_example.sav"
    model = pickle.load(open(filename,'rb'))
    comm.ml_ready()
    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.

    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()
    ball_position_history = []

    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()
        ball_position_history.append(scene_info.ball)
        Platform_center_x = scene_info.platform[0]+20
        if(len(ball_position_history) > 1):
            vx = ball_position_history[-1][0]-ball_position_history[-2][0]
            vy = ball_position_history[-1][1]-ball_position_history[-2][1]
            inp_temp=np.array([scene_info.ball[0], scene_info.ball[1], scene_info.platform[0],vx,vy])
            input = inp_temp[np.newaxis, :]
				
				
        if scene_info.status == GameStatus.GAME_OVER or scene_info.status == GameStatus.GAME_PASS:
            # Do some stuff if needed
            #scene_info = comm.get_scene_infso()
            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue
			
        if(len(ball_position_history) > 1):
            move=model.predict(input)
        else:
            move=0
        print(move)

        if move < 0:
            comm.send_instruction(scene_info.frame,PlatformAction.MOVE_LEFT)
        elif move > 0:
            comm.send_instruction(scene_info.frame,PlatformAction.MOVE_RIGHT)
        else :
            comm.send_instruction(scene_info.frame,PlatformAction.NONE)
			