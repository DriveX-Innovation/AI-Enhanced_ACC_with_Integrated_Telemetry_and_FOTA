import abc
import os
import numpy as np
import onnxruntime as ort


class EngineBase(abc.ABC):
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise Exception(f"Model path not found: {model_path}")
        assert model_path.endswith('.onnx'), "Only .onnx models are supported on Raspberry Pi"

        self._framework_type = None

    @property
    def framework_type(self):
        return self._framework_type

    @framework_type.setter
    def framework_type(self, value):
        self._framework_type = value

    @abc.abstractmethod
    def get_engine_input_shape(self):
        pass

    @abc.abstractmethod
    def get_engine_output_shape(self):
        pass

    @abc.abstractmethod
    def engine_inference(self, input_tensor):
        pass


class OnnxEngine(EngineBase):
    def __init__(self, onnx_file_path):
        super().__init__(onnx_file_path)

        # Force CPU execution on Raspberry Pi
        self.session = ort.InferenceSession(
            onnx_file_path,
            providers=["CPUExecutionProvider"]
        )

        self.providers = self.session.get_providers()
        self.engine_dtype = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
        self.framework_type = "onnx"

        self.__load_engine_interface()

    def __load_engine_interface(self):
        self.__input_shape = [inp.shape for inp in self.session.get_inputs()]
        self.__input_names = [inp.name for inp in self.session.get_inputs()]
        self.__output_shapes = [out.shape for out in self.session.get_outputs()]
        self.__output_names = [out.name for out in self.session.get_outputs()]

    def get_engine_input_shape(self):
        return self.__input_shape[0]

    def get_engine_output_shape(self):
        return self.__output_shapes, self.__output_names

    def engine_inference(self, input_tensor):
        outputs = self.session.run(
            self.__output_names,
            {self.__input_names[0]: input_tensor}
        )
        return outputs
