�	 ���6C@ ���6C@! ���6C@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails- ���6C@�����A@1�1 ��?A��&��?IP���(I@*	33333�F@2U
Iterator::Model::ParallelMapV2 �o_Ή?!ݔT���;@) �o_Ή?1ݔT���;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF��_�?!B7%�!6:@)��ׁsF�?1���`�5@:Preprocessing2F
Iterator::ModeljM�?!�3��	E@)9��v��z?1ȥoe�,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{�G�z�?!覤:�6@)-C��6z?1Sb�1,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!�`��ҷ@)��H�}m?1�`��ҷ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	�c�?!�y\��L@)a��+ei?1hGP@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!TRb�@)����Mb`?1TRb�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM�St$�?!��`��8@)��_�LU?1ӷ�2Q�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 93.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��ӟ��X@Q�,?��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�����A@�����A@!�����A@      ��!       "	�1 ��?�1 ��?!�1 ��?*      ��!       2	��&��?��&��?!��&��?:	P���(I@P���(I@!P���(I@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��ӟ��X@y�,?��?�"5
sequential/dense/MatMulMatMul�2	}ɟ�?!�2	}ɟ�?0"C
%gradient_tape/sequential/dense/MatMulMatMul?-��q�?!�����?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul?-��q�?!�ƌf{A�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul.O�ǆ�?!?-��q�?"7
sequential/dense_1/MatMulMatMul.O�ǆ�?!%�ɜoB�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�8
R�͕?!7>'/��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�8
R�͕?!I�L���?"!
Adam/PowPow?-��q�?!�M�!��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad?-��q�?!��N[T�?"7
sequential/dense_2/MatMulMatMul?-��q�?! �'�C��?0Q      Y@Y>����/@aX�i��U@qaMҗy�X@y��q�?"�
both�Your program is POTENTIALLY input-bound because 93.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 