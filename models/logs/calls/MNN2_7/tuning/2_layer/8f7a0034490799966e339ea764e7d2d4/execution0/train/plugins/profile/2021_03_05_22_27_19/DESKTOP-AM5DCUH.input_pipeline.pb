	��k&߈7@��k&߈7@!��k&߈7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��k&߈7@��nf��4@1�=�4a�?A�y��Q��?ID�.l�V@*	33333�G@2U
Iterator::Model::ParallelMapV29��v���?!/��m;@)9��v���?1/��m;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!�w��	�9@)��ZӼ�?1
�Z܄5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg��j+��?!^-n���8@)� �	�?1(�X�>0@:Preprocessing2F
Iterator::Model�j+��ݓ?!C����vD@)-C��6z?1����+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!l��Ӭ� @)����Mbp?1l��Ӭ� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy�&1��?!�ur.�M@)Ǻ���f?1��2��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!q�w��@)�J�4a?1q�w��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�{�Pk�?!n���7;@)a2U0*�S?1��td�@@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��b=U�X@Q��W����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��nf��4@��nf��4@!��nf��4@      ��!       "	�=�4a�?�=�4a�?!�=�4a�?*      ��!       2	�y��Q��?�y��Q��?!�y��Q��?:	D�.l�V@D�.l�V@!D�.l�V@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��b=U�X@y��W����?