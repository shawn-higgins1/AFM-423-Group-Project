	����G3@����G3@!����G3@	Õ4���?Õ4���?!Õ4���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6����G3@�H��1@1�M+�@.�?A�J�4�?Iu�BY�z�?Y�,��o��?*	     �G@2U
Iterator::Model::ParallelMapV2���QI�?!R�٨�l>@)���QI�?1R�٨�l>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!,����8@)n���?1���Q��4@:Preprocessing2F
Iterator::Model�g��s��?!�;���F@)lxz�,C|?11���\-@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!�+���6@)_�Q�{?1�Q�٨�,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!r1��� @)�q����o?1r1��� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�{�Pk�?!��
brK@)_�Q�k?1�Q�٨�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!!&W�+@)ŏ1w-!_?1!&W�+@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Õ4���?IN���L�X@Qx3���v�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�H��1@�H��1@!�H��1@      ��!       "	�M+�@.�?�M+�@.�?!�M+�@.�?*      ��!       2	�J�4�?�J�4�?!�J�4�?:	u�BY�z�?u�BY�z�?!u�BY�z�?B      ��!       J	�,��o��?�,��o��?!�,��o��?R      ��!       Z	�,��o��?�,��o��?!�,��o��?b      ��!       JGPUYÕ4���?b qN���L�X@yx3���v�?