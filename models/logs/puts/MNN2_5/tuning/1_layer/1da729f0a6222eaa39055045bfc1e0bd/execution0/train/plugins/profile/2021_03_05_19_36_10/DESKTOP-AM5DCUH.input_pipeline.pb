	W��:F5@W��:F5@!W��:F5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-W��:F5@�C3O��2@1"���/�?A�lV}��?I4�Op�� @*	������F@2U
Iterator::Model::ParallelMapV2�~j�t��?!�5��P:@)�~j�t��?1�5��P:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM�St$�?!~���8@)U���N@�?1�YLg1�4@:Preprocessing2F
Iterator::ModelU���N@�?!�YLg1�D@)_�Q�{?11��t�-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�I+��?!��Gp8@)S�!�uq{?1.����b-@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"��u��q?!�}��"@)"��u��q?1�}��"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�!�uq�?!.����bM@)a��+ei?1�YLg1@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!������@)ŏ1w-!_?1������@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa��+e�?!�YLg1;@)Ǻ���V?1��#��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���5�X@QCX���2�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�C3O��2@�C3O��2@!�C3O��2@      ��!       "	"���/�?"���/�?!"���/�?*      ��!       2	�lV}��?�lV}��?!�lV}��?:	4�Op�� @4�Op�� @!4�Op�� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���5�X@yCX���2�?