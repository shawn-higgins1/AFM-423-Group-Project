	�Z�Q6@�Z�Q6@!�Z�Q6@	$��Ǎ@�?$��Ǎ@�?!$��Ǎ@�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�Z�Q6@fl�f�3@1M1AG��?A�e��a��?I�'I�L> @Y7����q?*	effff&J@2U
Iterator::Model::ParallelMapV2�5�;Nё?!���3��@@)�5�;Nё?1���3��@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!_���.6@)�j+��݃?1���
��2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�{�Pk�?!�� W�8@)�&S��?1�|�VMf1@:Preprocessing2F
Iterator::Model�~j�t��?!�����F@)F%u�{?1�D�z/=)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!���('@)ŏ1w-!o?1���('@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipB>�٬��?!VMf�1K@)�����g?1_���.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!���('@)ŏ1w-!_?1���('@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaplxz�,C�?!v��.�b:@)��H�}M?1e�㐈�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9$��Ǎ@�?IUܓQ�X@Qa�Gz5�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	fl�f�3@fl�f�3@!fl�f�3@      ��!       "	M1AG��?M1AG��?!M1AG��?*      ��!       2	�e��a��?�e��a��?!�e��a��?:	�'I�L> @�'I�L> @!�'I�L> @B      ��!       J	7����q?7����q?!7����q?R      ��!       Z	7����q?7����q?!7����q?b      ��!       JGPUY$��Ǎ@�?b qUܓQ�X@ya�Gz5�?