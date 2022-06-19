import os
import pybullet as p
import pybullet_data


def init_simulation(render = False):

    if render:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.setTimeStep(0.01)

    '------------------------------------'
    # drone
    drone = p.loadURDF(os.path.join(os.getcwd(),'urdf/drone.urdf'), 
                        baseOrientation= p.getQuaternionFromEuler([0,0,0]))

    # marker at desired point
    sphereVisualId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                        radius = 0.05,
                                        rgbaColor= [1, 0, 0, 1])
    marker = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=sphereVisualId, basePosition=[0, 0, 5],
                    baseOrientation=p.getQuaternionFromEuler([0,0,0]),
                    useMaximalCoordinates=False)
    '-------------------------------------'

    p.resetDebugVisualizerCamera( cameraDistance=3.5, cameraYaw=30, 
                                cameraPitch=-20, cameraTargetPosition=[0,0,5])

    return drone, marker

def end_simulation():
    p.disconnect()