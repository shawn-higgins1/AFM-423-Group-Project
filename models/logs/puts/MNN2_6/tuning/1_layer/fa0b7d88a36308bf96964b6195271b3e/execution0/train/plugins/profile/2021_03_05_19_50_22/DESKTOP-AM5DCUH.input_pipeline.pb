	1�~�ٱ5@1�~�ٱ5@!1�~�ٱ5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-1�~�ٱ5@�.5B?C3@1�s�f���?A|�Pk��?I�8�� n@*	������F@2U
Iterator::Model::ParallelMapV2�������?!��b:�i;@)�������?1��b:�i;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!��P^C�:@)��ZӼ�?1�5��P^6@:Preprocessing2F
Iterator::Modela2U0*��?!�P^CyE@)S�!�uq{?1.����b-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea2U0*��?!�P^Cy5@)�I+�v?1��Gp(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!���>��!@)	�^)�p?1���>��!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipF%u��?!)�����L@)��_vOf?1��k(�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!����k@)�J�4a?1����k@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�+e�X�?!      9@)��H�}]?1Cy�5�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI:�%���X@Qc:m/��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�.5B?C3@�.5B?C3@!�.5B?C3@      ��!       "	�s�f���?�s�f���?!�s�f���?*      ��!       2	|�Pk��?|�Pk��?!|�Pk��?:	�8�� n@�8�� n@!�8�� n@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q:�%���X@yc:m/��?