	���o�1@���o�1@!���o�1@	�����?�����?!�����?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���o�1@�e��ay0@1������?Ash��|?�?I��9x&��?YM֨�h�?*	33333�I@2U
Iterator::Model::ParallelMapV2lxz�,C�?!'�&�&�:@)lxz�,C�?1'�&�&�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�!�uq�?!���:@)Ǻ����?16�5�5�5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���S㥋?!�C�C�C:@)Έ����?1���2@:Preprocessing2F
Iterator::Model^K�=��?!y�y�y�D@)�<,Ԛ�}?1�g�g�g,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�J�4q?!�W�W�W @)�J�4q?1�W�W�W @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�sF���?!�y�y�yM@)_�Q�k?1�u�u�u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!���@)/n��b?1���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�����?I�U�U�X@QH7`6y�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�e��ay0@�e��ay0@!�e��ay0@      ��!       "	������?������?!������?*      ��!       2	sh��|?�?sh��|?�?!sh��|?�?:	��9x&��?��9x&��?!��9x&��?B      ��!       J	M֨�h�?M֨�h�?!M֨�h�?R      ��!       Z	M֨�h�?M֨�h�?!M֨�h�?b      ��!       JGPUY�����?b q�U�U�X@yH7`6y�?