	H��'�"7@H��'�"7@!H��'�"7@	j�Ԉɚ?j�Ԉɚ?!j�Ԉɚ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6H��'�"7@��Pj/�4@1�k	��g�?A��0�*�?I��	h"L@Y>�4a��x?*	    @H@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate���&�?!���AGC@)"��u���?1Z�D�a�A@:Preprocessing2U
Iterator::Model::ParallelMapV2�(��0�?!���_\9@)�(��0�?1���_\9@:Preprocessing2F
Iterator::Modela2U0*��?!��4l7�C@)lxz�,C|?1T���t,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+ey?!i�n�'�)@)����Mbp?1�Q�/�~ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�X�� �?!l7˓�4N@)-C��6j?1���Id@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!2���$@)/n��b?12���$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap8��d�`�?!�
|q��D@)a2U0*�S?1��4l7�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!�,O"Ӱ�?)��H�}M?1�,O"Ӱ�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!��4l7��?)a2U0*�C?1��4l7��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9j�Ԉɚ?IιI�k�X@Q�8n��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Pj/�4@��Pj/�4@!��Pj/�4@      ��!       "	�k	��g�?�k	��g�?!�k	��g�?*      ��!       2	��0�*�?��0�*�?!��0�*�?:	��	h"L@��	h"L@!��	h"L@B      ��!       J	>�4a��x?>�4a��x?!>�4a��x?R      ��!       Z	>�4a��x?>�4a��x?!>�4a��x?b      ��!       JGPUYj�Ԉɚ?b qιI�k�X@y�8n��?