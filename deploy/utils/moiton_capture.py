import numpy as np
import logging
from nokov.nokovsdk import *


class MotionCapture:

    def __init__(self):
        SERVER_IP = "192.168.200.154"
        self.logger = logging.getLogger(__name__)
        self.init = True
        self.client = PySDKClient()
        self.logger.info("Begin to init the SDK Client")
        ret = self.client.Initialize(bytes(SERVER_IP, encoding="utf8"))
        if ret == 0:
            self.logger.info("Connect to the Nokov Succeed")
        else:
            self.logger.error(f"Connect Failed: {ret}")
            raise ConnectionError("Failed to connect to the Nokov server.")
        self.base_pos = np.zeros(2)
        self.ball_pos = np.zeros(2)
        self.preFrmNo = None
        self.robot_id = self._find_robot_id()
        self.client.PySetDataCallback(self.py_data_func, None)

    def _find_robot_id(self):
        pdds = POINTER(DataDescriptions)()
        self.client.PyGetDataDescriptions(pdds)
        dataDefs = pdds.contents
        for iDef in range(dataDefs.nDataDescriptions):
            dataDef = dataDefs.arrDataDescriptions[iDef]
            if dataDef.type == DataDescriptors.Descriptor_RigidBody.value:
                rigidBody = dataDef.Data.RigidBodyDescription.contents
                if "Robot" in rigidBody.szName.decode("utf-8"):
                    robot_id = rigidBody.ID
                    self.logger.info(f"robot_id: {robot_id}")
                    return robot_id
        self.logger.warning("Robot ID not found.")
        return None

    def py_data_func(self, pFrameOfMocapData, pUserData):
        if pFrameOfMocapData is None:
            self.logger.warning("Not get the data frame.")
            return

        frameData = pFrameOfMocapData.contents
        if frameData.iFrame == self.preFrmNo:
            return
        self.preFrmNo = frameData.iFrame

        length = 128
        szTimeCode = bytes(length)
        self.client.PyTimecodeStringify(frameData.Timecode, frameData.TimecodeSubframe, szTimeCode, length)

        self._update_base_pos(frameData)
        self._update_ball_pos(frameData)

        if self.init:
            self.logger.info(f"Init Robot Pos: {self.base_pos}")
            self.logger.info(f"Init Ball Pos: {self.ball_pos}")
            self.init = False

    def _update_base_pos(self, frameData):
        for i in range(frameData.nRigidBodies):
            body = frameData.RigidBodies[i]
            if abs(body.x) < 20000 and abs(body.y) < 20000:
                if body.ID == self.robot_id:
                    self.base_pos[0] = body.x * 1.0e-3
                    self.base_pos[1] = body.y * 1.0e-3

    def _update_ball_pos(self, frameData):
        for i in range(frameData.nOtherMarkers):
            otherMarker = frameData.OtherMarkers[i]
            if otherMarker[2] < 300:
                self.ball_pos[0] = otherMarker[0] * 1.0e-3
                self.ball_pos[1] = otherMarker[1] * 1.0e-3
