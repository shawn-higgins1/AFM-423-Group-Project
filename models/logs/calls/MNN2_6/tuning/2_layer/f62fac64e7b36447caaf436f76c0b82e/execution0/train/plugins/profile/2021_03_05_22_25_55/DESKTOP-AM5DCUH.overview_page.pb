�	��M���4@��M���4@!��M���4@	�U��#��?�U��#��?!�U��#��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��M���4@B��=2@1	�3���?A�J�4�?IW�9�mb@Y�'����?*	ffffffM@2U
Iterator::Model::ParallelMapV2	�^)ː?!�kv�"�;@)	�^)ː?1�kv�"�;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�N@aÓ?!�7�L\i@@)K�=�U�?1��)x9:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz�,C�?!^N��)x7@)g��j+��?19/���3@:Preprocessing2F
Iterator::ModelǺ���?!�0�0C@)�~j�t�x?1�Cc}h$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!ΎZ��5@)����Mbp?1ΎZ��5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�&S��?!=��<��N@)F%u�k?1���S�r@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!$I�$I�@)�J�4a?1$I�$I�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9 V��#��?IB��s��X@Q,��4��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	B��=2@B��=2@!B��=2@      ��!       "		�3���?	�3���?!	�3���?*      ��!       2	�J�4�?�J�4�?!�J�4�?:	W�9�mb@W�9�mb@!W�9�mb@B      ��!       J	�'����?�'����?!�'����?R      ��!       Z	�'����?�'����?!�'����?b      ��!       JGPUY V��#��?b qB��s��X@y,��4��?�"5
sequential/dense/MatMulMatMul�㰽��?!�㰽��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�����?!���_AG�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�����?!@��a�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�����?!,�z7���?"7
sequential/dense_1/MatMulMatMul�Y"m�X�?!�[�R�5�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�[�R�5�?!�m|�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�P�>��?!�XvT\�?"7
sequential/dense_2/MatMulMatMulG6��G��?!��|H��?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�����?!�F���D�?"E
'gradient_tape/sequential/dense_2/MatMulMatMul1���P�?!��8<ŉ�?0Q      Y@Y7��Moz2@a���,daT@q� �	G�V@y@��a�?"�
both�Your program is POTENTIALLY input-bound because 86.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�91.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 