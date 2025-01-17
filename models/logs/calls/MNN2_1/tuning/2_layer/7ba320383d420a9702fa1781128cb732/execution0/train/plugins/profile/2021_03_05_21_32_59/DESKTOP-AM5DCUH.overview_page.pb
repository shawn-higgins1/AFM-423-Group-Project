�	�^��7@�^��7@!�^��7@	1��hB�?1��hB�?!1��hB�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�^��7@V+~��4@1H����p�?A,e�X�?IG�tF�@Y/3l��{?*	�����G@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateX�5�;N�?!)�-fhTB@)X9��v��?1~����@@:Preprocessing2U
Iterator::Model::ParallelMapV2�HP��?!�Y�v:@)�HP��?1�Y�v:@:Preprocessing2F
Iterator::Model���&�?!P5 �0HD@)9��v��z?1Ӕ�3,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v��z?!Ӕ�3,@)U���N@s?1n"���c$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?�߾�?!���!ϷM@)Ǻ���f?1�z�ԅK@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!\�*�<@)��H�}]?1\�*�<@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�:pΈ�?!��$���C@)a2U0*�S?1��q��@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!�����?)-C��6J?1�����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!��q���?)a2U0*�C?1��q���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no91��hB�?I���}]�X@Q!Ri�s�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	V+~��4@V+~��4@!V+~��4@      ��!       "	H����p�?H����p�?!H����p�?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	G�tF�@G�tF�@!G�tF�@B      ��!       J	/3l��{?/3l��{?!/3l��{?R      ��!       Z	/3l��{?/3l��{?!/3l��{?b      ��!       JGPUY1��hB�?b q���}]�X@y!Ri�s�?�"5
sequential/dense/MatMulMatMul篴ƽ5�?!篴ƽ5�?0"C
%gradient_tape/sequential/dense/MatMulMatMulGK[S�?!v����$�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulGK[S�?!���>2��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulGK[S�?!>�%���?"7
sequential/dense_1/MatMulMatMul���@�?!.�!�}��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�2(X�?!Gce����?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�2(X�?!`&��2�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradGK[S�?!A�p�T�?"7
sequential/dense_2/MatMulMatMulGK[S�?!"�{�\w�?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamx���,�?!G9���?Q      Y@Y�C=�C=0@a��
��T@q�e�W@y���>2��?"�
both�Your program is POTENTIALLY input-bound because 86.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 