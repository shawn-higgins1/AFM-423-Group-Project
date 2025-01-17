�		�v�3@	�v�3@!	�v�3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-	�v�3@��8ӄ)2@1-AF@�#�?A�a��4�?I���h8�?*	433333L@2U
Iterator::Model::ParallelMapV2� �	��?!�y�'N;@)� �	��?1�y�'N;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapjM�?!>�2t��@@)� �	��?1�y�'N;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!4/�D�)6@)�0�*�?1��c-C2@:Preprocessing2F
Iterator::Model��JY�8�?!�O$��<C@) �o_�y?1AL� &W&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!Y���=@)���_vOn?1Y���=@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��W�2ġ?!��c-�N@)a��+ei?1(Ni^�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!��	�4@)/n��b?1��	�4@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI �sw�X@Qx#"F�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��8ӄ)2@��8ӄ)2@!��8ӄ)2@      ��!       "	-AF@�#�?-AF@�#�?!-AF@�#�?*      ��!       2	�a��4�?�a��4�?!�a��4�?:	���h8�?���h8�?!���h8�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q �sw�X@yx#"F�?�"5
sequential/dense/MatMulMatMul8M)�w�?!8M)�w�?0"C
%gradient_tape/sequential/dense/MatMulMatMulL����?!�o��1�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�~g4�`�?!���1�?"7
sequential/dense_1/MatMulMatMul`��4�_�?!m��aI�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul xo�?!n�?�L�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad8M)�w�?!?��{�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul8M)�w�?!^�)�V��?"7
sequential/dense_2/MatMulMatMul8M)�w�?!2IL0�l�?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam`��4�_�?!�&�#Բ�?"E
'gradient_tape/sequential/dense_2/MatMulMatMul`��4�_�?!����?0Q      Y@Y7��Moz2@a���,daT@qXtyLn�X@y�������?"�
both�Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 