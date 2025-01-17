�	S�!�u�1@S�!�u�1@!S�!�u�1@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-S�!�u�1@#,*�t~0@1����b�?A�o��}�?I�s����?*	�����F@2U
Iterator::Model::ParallelMapV2-C��6�?!B�O�b=@)-C��6�?1B�O�b=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�+e�X�?!w�"��9@)a2U0*��?1q�{��5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�g��s��?!�-�I	8@)�ZӼ�}?1>��0@:Preprocessing2F
Iterator::ModelΈ����?!F�_��E@)�����w?1��n�M*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�&1��?!�7�L\�L@)y�&1�l?1������@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!������@)y�&1�l?1������@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!����S@)��H�}]?1����S@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIt�O�
�X@Q�E,X���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	#,*�t~0@#,*�t~0@!#,*�t~0@      ��!       "	����b�?����b�?!����b�?*      ��!       2	�o��}�?�o��}�?!�o��}�?:	�s����?�s����?!�s����?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qt�O�
�X@y�E,X���?�"5
sequential/dense/MatMulMatMul��ZX!>�?!��ZX!>�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��$��?!"~�>٬�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��^Q�V�?!�b����?"7
sequential/dense_1/MatMulMatMul�,bza�?!8���?��?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradz��m�?!���m���?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdame��N��?!�+twٻ�?"!
Adam/PowPowe��N��?!!aE�2��?"E
'gradient_tape/sequential/dense_1/MatMulMatMule��N��?!'K��E��?0")
sequential/CastCaste��N��?!��sJ��?"
Abs_1Abs?�:Ԓ?!�Y"�3@�?Q      Y@Y�M�_{4@a��(�S@q�g��ˆX@yW�[}��?"�
both�Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 