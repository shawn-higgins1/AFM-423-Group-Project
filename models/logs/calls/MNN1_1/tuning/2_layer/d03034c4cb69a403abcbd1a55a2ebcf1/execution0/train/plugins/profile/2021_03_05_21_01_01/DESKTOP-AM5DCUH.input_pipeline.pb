	�;��7@�;��7@!�;��7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�;��7@��Q,��4@1�`��p�?A�HP��?I E���D@*	������O@2U
Iterator::Model::ParallelMapV28��d�`�?!N��J?@)8��d�`�?1N��J?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�ݓ��Z�?!�R��=@)L7�A`�?1e�Cj��9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�������?!΀Pr�3@)�� �rh�?1��{mĺ*@:Preprocessing2F
Iterator::Model�#�����?!Dj��VvE@)���_vO~?1q��;E'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!<Eg@(@)����Mbp?1<Eg@(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����镢?!��}��L@)F%u�k?1���!5�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!��sHM0@)a2U0*�c?1��sHM0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�!��u��?!�!5�x+6@)-C��6Z?1ʝ��3 @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�12.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIX�bM`�X@Q�A��g�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Q,��4@��Q,��4@!��Q,��4@      ��!       "	�`��p�?�`��p�?!�`��p�?*      ��!       2	�HP��?�HP��?!�HP��?:	 E���D@ E���D@! E���D@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qX�bM`�X@y�A��g�?