	]�C��'7@]�C��'7@!]�C��'7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-]�C��'7@�)��z�4@1�&�5��?A_)�Ǻ�?IA�Ρ� @*	������G@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate������?!�A�IB@)�q����?1��c+��@@:Preprocessing2U
Iterator::Model::ParallelMapV2�~j�t��?!4��}yl9@)�~j�t��?14��}yl9@:Preprocessing2F
Iterator::Modela2U0*��?!�Iݗ�VD@)��H�}}?1���c+�.@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6z?!'u_+@)U���N@s?1��N�#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy�&1��?!<�"h8�M@)F%u�k?1lE�pR�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!�}ylE�@)_�Q�[?1�}ylE�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<��?!N�<�bC@)a2U0*�S?1�Iݗ�V@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!���c+��?)��H�}M?1���c+��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�Iݗ�V�?)a2U0*�C?1�Iݗ�V�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���ƴ�X@Q�XΒ�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�)��z�4@�)��z�4@!�)��z�4@      ��!       "	�&�5��?�&�5��?!�&�5��?*      ��!       2	_)�Ǻ�?_)�Ǻ�?!_)�Ǻ�?:	A�Ρ� @A�Ρ� @!A�Ρ� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���ƴ�X@y�XΒ�?