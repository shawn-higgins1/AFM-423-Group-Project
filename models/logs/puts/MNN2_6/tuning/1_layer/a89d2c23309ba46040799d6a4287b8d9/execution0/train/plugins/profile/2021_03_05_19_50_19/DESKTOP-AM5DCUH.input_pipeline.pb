	��$�g5@��$�g5@!��$�g5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��$�g5@���Xl�2@1P�mp��?Ag��j+��?IYO���@*	      G@2U
Iterator::Model::ParallelMapV2S�!�uq�?!���,d!=@)S�!�uq�?1���,d!=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!���,d;@)/�$��?1��Moz�6@:Preprocessing2F
Iterator::Model8��d�`�?!���,d�E@)9��v��z?1d!Y�B,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��~j�t�?!zӛ���4@)Ǻ���v?1�,d!Y(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!ozӛ�� @)�q����o?1ozӛ�� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}гY���?!Oozӛ^L@)��_vOf?18��Moz@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!d!Y�B@)�J�4a?1d!Y�B@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_vO�?!8��Moz7@)��_�LU?1�Mozӛ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIT����X@QV��<�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���Xl�2@���Xl�2@!���Xl�2@      ��!       "	P�mp��?P�mp��?!P�mp��?*      ��!       2	g��j+��?g��j+��?!g��j+��?:	YO���@YO���@!YO���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qT����X@yV��<�?