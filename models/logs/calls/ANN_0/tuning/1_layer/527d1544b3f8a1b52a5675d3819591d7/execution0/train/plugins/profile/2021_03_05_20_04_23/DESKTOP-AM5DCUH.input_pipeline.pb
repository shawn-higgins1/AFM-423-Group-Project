	�磌�(3@�磌�(3@!�磌�(3@	���Z�R�?���Z�R�?!���Z�R�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�磌�(3@�4}v��0@1Ov3��?A��ͪ�զ?I�q6@Y!#���r?*	������P@2U
Iterator::Model::ParallelMapV2��ׁsF�?!6U��)=@)��ׁsF�?16U��)=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�� �rh�?!� ��l	9@)%u��?1b�Y�D�5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapr�����?!���W:@)�HP��?1�oQ���1@:Preprocessing2F
Iterator::Model�v��/�?!ϭ�U��D@)�5�;Nс?1��R<�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�I+�v?!�K�F3 @)�I+�v?1�K�F3 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���<,�?!2Rj�dM@)U���N@s?17��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!�*»B@)HP�s�b?1�*»B@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���Z�R�?I{"hWZ�X@Qc	=��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�4}v��0@�4}v��0@!�4}v��0@      ��!       "	Ov3��?Ov3��?!Ov3��?*      ��!       2	��ͪ�զ?��ͪ�զ?!��ͪ�զ?:	�q6@�q6@!�q6@B      ��!       J	!#���r?!#���r?!!#���r?R      ��!       Z	!#���r?!#���r?!!#���r?b      ��!       JGPUY���Z�R�?b q{"hWZ�X@yc	=��?