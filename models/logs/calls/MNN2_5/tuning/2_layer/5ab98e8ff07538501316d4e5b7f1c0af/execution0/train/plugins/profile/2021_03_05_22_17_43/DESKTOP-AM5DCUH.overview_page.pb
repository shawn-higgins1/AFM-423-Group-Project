�	�W˝��3@�W˝��3@!�W˝��3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�W˝��3@�O ��"2@1����c�?A�@��ǘ�?I��*�]'�?*	effff&G@2U
Iterator::Model::ParallelMapV2V-��?!�}SGQ?@)V-��?1�}SGQ?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!�51��9@)n���?1 Z�*5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_vO�?!b�
�}S7@)y�&1�|?1bR��<.@:Preprocessing2F
Iterator::Model/�$��?!������F@)9��v��z?1����,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceŏ1w-!o?!a�o�(j @)ŏ1w-!o?1a�o�(j @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipc�ZB>�?!6DkbRK@)�����g?1�51��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�����@)��H�}]?1�����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIrY��X@Qy���;8�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�O ��"2@�O ��"2@!�O ��"2@      ��!       "	����c�?����c�?!����c�?*      ��!       2	�@��ǘ�?�@��ǘ�?!�@��ǘ�?:	��*�]'�?��*�]'�?!��*�]'�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qrY��X@yy���;8�?�"C
%gradient_tape/sequential/dense/MatMulMatMulc���3�?!c���3�?0"5
sequential/dense/MatMulMatMulc���3�?!c���3�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul��`Yퟤ?!������?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��`Yퟤ?!���.���?"7
sequential/dense_1/MatMulMatMul��`Yퟤ?!�"O���?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad���Ǚ?!�EF��J�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul���Ǚ?!������?"7
sequential/dense_2/MatMulMatMul�Ϡ)L�?!��j�V�?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam��`Yퟔ?!��.@���?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�L��?!YY_`1��?Q      Y@Y7��Moz2@a���,daT@q�g@��|X@y/��m��?"�
both�Your program is POTENTIALLY input-bound because 91.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 