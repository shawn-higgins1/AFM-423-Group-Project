	2���"4@2���"4@!2���"4@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-2���"4@�O ��r2@1��	m9�?A0*��D�?I�0�d�?*	     H@2U
Iterator::Model::ParallelMapV2���_vO�?!VUUUU�>@)���_vO�?1VUUUU�>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!     `8@)��~j�t�?1������3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!     `8@)ŏ1w-!?1������/@:Preprocessing2F
Iterator::Modelj�t��?!VUUUUeF@)S�!�uq{?1������+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice	�^)�p?!UUUUU!@)	�^)�p?1UUUUU!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�A`��"�?!������K@)a��+ei?1VUUUU�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!UUUUUU@)/n��b?1UUUUUU@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI ��8�xX@Q\��� @Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�O ��r2@�O ��r2@!�O ��r2@      ��!       "	��	m9�?��	m9�?!��	m9�?*      ��!       2	0*��D�?0*��D�?!0*��D�?:	�0�d�?�0�d�?!�0�d�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q ��8�xX@y\��� @